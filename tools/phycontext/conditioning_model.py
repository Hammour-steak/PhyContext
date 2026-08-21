from pathlib import Path

import numpy as np
import torch
from torch import nn

from point_trajectory import MAX_OBJECTS


def load_scene_condition(path: Path, device=None) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        result = {
            "object_xyz_camera_m": torch.from_numpy(
                archive["object_xyz_camera_m"].astype(np.float32)
            ),
            "object_normal_camera": torch.from_numpy(
                archive["object_normal_camera"].astype(np.float32)
            ),
            "environment_xyz_camera_m": torch.from_numpy(
                archive["environment_xyz_camera_m"].astype(np.float32)
            ),
            "environment_normal_camera": torch.from_numpy(
                archive["environment_normal_camera"].astype(np.float32)
            ),
            "environment_friction": torch.from_numpy(
                archive["environment_friction"].astype(np.float32)
            ),
            "environment_restitution": torch.from_numpy(
                archive["environment_restitution"].astype(np.float32)
            ),
            "camera_intrinsics_normalized": torch.from_numpy(
                archive["camera_intrinsics_normalized"].astype(np.float32)
            ),
        }
    if device is not None:
        result = {key: value.to(device) for key, value in result.items()}
    return result


def collate_scene_conditions(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("cannot collate an empty scene batch")
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def _dynamic_objects(record: dict) -> list[dict]:
    physics = record["conditioning"]["physics"]
    objects = physics.get("objects")
    if objects is None:
        return [physics["object"]]
    if isinstance(objects, dict):
        return [objects[key] for key in sorted(objects)]
    if not isinstance(objects, list) or not objects:
        raise ValueError("physics.objects must be a non-empty list or mapping")
    return objects


def controls_from_record(record: dict, device=None) -> torch.Tensor:
    values = torch.tensor(
        [
            [item["mass_kg"], item["friction"], item["restitution"]]
            for item in _dynamic_objects(record)
        ],
        dtype=torch.float32,
        device=device,
    )
    return values[0] if values.shape[0] == 1 else values


def dynamics_from_record(record: dict, device=None) -> torch.Tensor:
    values = []
    for dynamic_object in _dynamic_objects(record):
        inertia = dynamic_object["inertia_tensor_camera_kg_m2"]
        values.append(
            [
                inertia[0][0],
                inertia[1][1],
                inertia[2][2],
                inertia[0][1],
                inertia[0][2],
                inertia[1][2],
                dynamic_object["rolling_friction"],
                dynamic_object["spinning_friction"],
                dynamic_object["linear_damping"],
                dynamic_object["angular_damping"],
            ]
        )
    values = torch.tensor(
        values,
        dtype=torch.float32,
        device=device,
    )
    return values[0] if values.shape[0] == 1 else values


def initial_state_from_record(record: dict, device=None) -> torch.Tensor:
    physics = record["conditioning"]["physics"]
    values = torch.tensor(
        [
            [
                *item["initial_state"]["linear_velocity_camera_m_s"],
                *item["initial_state"]["angular_velocity_camera_rad_s"],
                *physics["world"]["gravity_camera_m_s2"],
            ]
            for item in _dynamic_objects(record)
        ],
        dtype=torch.float32,
        device=device,
    )
    return values[0] if values.shape[0] == 1 else values


def apply_condition_mode(
    condition_tokens: torch.Tensor,
    scene_token_count: int,
    mode: str,
) -> torch.Tensor:
    """Mask trained condition slots for controlled inference ablations."""
    if condition_tokens.ndim != 3:
        raise ValueError("condition tokens must have shape B x L x C")
    if not 0 < scene_token_count < condition_tokens.shape[1]:
        raise ValueError("scene token count must split scene and physics tokens")
    if mode == "full":
        return condition_tokens
    masked = condition_tokens.clone()
    if mode == "scene_only":
        masked[:, scene_token_count:] = 0
    elif mode == "physics_only":
        masked[:, :scene_token_count] = 0
    elif mode == "adapter_only":
        masked.zero_()
    else:
        raise ValueError(f"unsupported condition mode: {mode}")
    return masked


class SceneCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.point_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, queries, points):
        normalized_queries = self.query_norm(queries)
        normalized_points = self.point_norm(points)
        attended, _ = self.attention(
            normalized_queries,
            normalized_points,
            normalized_points,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class ScenePointEncoder(nn.Module):
    """Compress fixed scene points into a fixed number of Wan context tokens."""

    def __init__(
        self,
        hidden_dim: int = 256,
        context_dim: int = 4096,
        token_count: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        categorical_dim: int = 8,
    ):
        super().__init__()
        self.token_count = token_count
        self.body_embedding = nn.Embedding(2, categorical_dim)
        self.object_slot_embedding = nn.Embedding(MAX_OBJECTS, hidden_dim)
        nn.init.zeros_(self.object_slot_embedding.weight)
        numeric_dim = 3 + 3 + 2 + 4
        self.point_projection = nn.Sequential(
            nn.Linear(numeric_dim + categorical_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Keep common parameter initialization identical across token-count
        # ablations. Each size takes a prefix from the same per-run query bank
        # without advancing PyTorch's global RNG state.
        query_generator = torch.Generator(device="cpu")
        query_generator.manual_seed(torch.initial_seed() ^ 0x5343454E45)
        self.queries = nn.Parameter(
            torch.randn(
                token_count,
                hidden_dim,
                generator=query_generator,
            )
            * 0.02
        )
        self.blocks = nn.ModuleList(
            [SceneCrossAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, context_dim))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, scene: dict[str, torch.Tensor]) -> torch.Tensor:
        object_xyz = scene["object_xyz_camera_m"]
        environment_xyz = scene["environment_xyz_camera_m"]
        object_normal = scene["object_normal_camera"]
        environment_normal = scene["environment_normal_camera"]
        if environment_xyz.ndim == 2:
            if object_xyz.ndim == 2:
                object_xyz = object_xyz.unsqueeze(0).unsqueeze(0)
                object_normal = object_normal.unsqueeze(0).unsqueeze(0)
            elif object_xyz.ndim == 3:
                object_xyz = object_xyz.unsqueeze(0)
                object_normal = object_normal.unsqueeze(0)
            else:
                raise ValueError("unbatched object points must have shape O x N x 3")
            environment_xyz = environment_xyz.unsqueeze(0)
            environment_normal = environment_normal.unsqueeze(0)
            environment_friction = scene["environment_friction"].unsqueeze(0)
            environment_restitution = scene["environment_restitution"].unsqueeze(0)
            camera_intrinsics = scene["camera_intrinsics_normalized"].unsqueeze(0)
        else:
            if object_xyz.ndim == 3:
                object_xyz = object_xyz.unsqueeze(1)
                object_normal = object_normal.unsqueeze(1)
            elif object_xyz.ndim != 4:
                raise ValueError("batched object points must have shape B x O x N x 3")
            if environment_xyz.ndim != 3:
                raise ValueError("environment points must have shape B x N x 3")
            environment_friction = scene["environment_friction"]
            environment_restitution = scene["environment_restitution"]
            camera_intrinsics = scene["camera_intrinsics_normalized"]
        batch, object_count, object_point_count, _ = object_xyz.shape
        if object_count > MAX_OBJECTS:
            raise ValueError(
                f"scene has {object_count} dynamic objects; maximum is {MAX_OBJECTS}"
            )
        environment_count = environment_xyz.shape[1]
        object_xyz = object_xyz.reshape(batch, object_count * object_point_count, 3)
        object_normal = object_normal.reshape(batch, object_count * object_point_count, 3)
        camera = camera_intrinsics.float()
        object_camera = camera.unsqueeze(1).expand(
            batch, object_count * object_point_count, 4
        )
        environment_camera = camera.unsqueeze(1).expand(
            batch, environment_count, 4
        )
        object_numeric = torch.cat(
            [
                torch.asinh(object_xyz.float()),
                object_normal.float(),
                torch.zeros(
                    batch,
                    object_count * object_point_count,
                    2,
                    dtype=object_xyz.dtype,
                    device=object_xyz.device,
                ),
                object_camera,
            ],
            dim=-1,
        )
        environment_numeric = torch.cat(
            [
                torch.asinh(environment_xyz.float()),
                environment_normal.float(),
                torch.cat(
                    [
                        torch.log1p(
                            environment_friction.float().clamp_min(0.0)
                        ),
                        environment_restitution.float().clamp(0.0, 1.0),
                    ],
                    dim=-1,
                ),
                environment_camera,
            ],
            dim=-1,
        )
        object_body = self.body_embedding(
            torch.ones(
                batch,
                object_count * object_point_count,
                dtype=torch.long,
                device=object_xyz.device,
            )
        )
        environment_body = self.body_embedding(
            torch.zeros(
                batch,
                environment_count,
                dtype=torch.long,
                device=object_xyz.device,
            )
        )
        object_points = self.point_projection(
            torch.cat([object_numeric, object_body], dim=-1)
        )
        object_slot_ids = torch.arange(
            object_count, device=object_xyz.device
        ).repeat_interleave(object_point_count)
        object_points = object_points + self.object_slot_embedding(
            object_slot_ids
        ).unsqueeze(0)
        environment_points = self.point_projection(
            torch.cat([environment_numeric, environment_body], dim=-1)
        )
        points = torch.cat([object_points, environment_points], dim=1)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            queries = block(queries, points)
        return self.output(queries)


class PhysicsControlEncoder(nn.Module):
    """Encode object mass, friction, and restitution as three sweep tokens."""

    def __init__(self, hidden_dim: int = 256, context_dim: int = 4096):
        super().__init__()
        self.parameter_embedding = nn.Embedding(3, hidden_dim)
        self.value_projection = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, context_dim)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        if controls.ndim == 1:
            controls = controls.unsqueeze(0)
        if controls.ndim not in {2, 3} or controls.shape[-1] != 3:
            raise ValueError("controls must contain mass, friction, and restitution")
        object_shape = controls.shape[:-1]
        flat_controls = controls.reshape(-1, 3)
        normalized = flat_controls.float().clone()
        normalized[:, 0] = torch.log(normalized[:, 0].clamp_min(1e-6))
        normalized[:, 1] = torch.log1p(normalized[:, 1].clamp_min(0.0))
        normalized[:, 2] = normalized[:, 2].clamp(0.0, 1.0)
        types = self.parameter_embedding(
            torch.arange(3, device=flat_controls.device)
        ).unsqueeze(0)
        hidden = self.value_projection(normalized.unsqueeze(-1)) + types
        tokens = self.output(hidden)
        if len(object_shape) == 2:
            return tokens.reshape(object_shape[0], object_shape[1] * 3, -1)
        return tokens


class InitialStateEncoder(nn.Module):
    """Encode initial linear velocity, angular velocity, and gravity."""

    def __init__(self, hidden_dim: int = 256, context_dim: int = 4096):
        super().__init__()
        self.state_embedding = nn.Embedding(3, hidden_dim)
        self.vector_projection = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, context_dim)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim not in {2, 3} or state.shape[-1] != 9:
            raise ValueError(
                "initial state must contain linear velocity, angular velocity, and gravity"
            )
        object_shape = state.shape[:-1]
        flat_state = state.reshape(-1, 9)
        vectors = flat_state.float().reshape(flat_state.shape[0], 3, 3).clone()
        vectors[:, 0] = torch.asinh(vectors[:, 0])
        vectors[:, 1] = torch.asinh(vectors[:, 1] / 5.0)
        vectors[:, 2] = vectors[:, 2] / 9.81
        types = self.state_embedding(torch.arange(3, device=flat_state.device)).unsqueeze(0)
        tokens = self.output(self.vector_projection(vectors) + types)
        if len(object_shape) == 2:
            return tokens.reshape(object_shape[0], object_shape[1] * 3, -1)
        return tokens


class BaseDynamicsEncoder(nn.Module):
    """Encode inertia, rolling/spinning friction, and damping as ten tokens."""

    def __init__(self, hidden_dim: int = 256, context_dim: int = 4096):
        super().__init__()
        self.parameter_embedding = nn.Embedding(10, hidden_dim)
        self.value_projection = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, context_dim)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, dynamics: torch.Tensor) -> torch.Tensor:
        if dynamics.ndim == 1:
            dynamics = dynamics.unsqueeze(0)
        if dynamics.ndim not in {2, 3} or dynamics.shape[-1] != 10:
            raise ValueError("base dynamics must contain six inertia and four loss terms")
        object_shape = dynamics.shape[:-1]
        flat_dynamics = dynamics.reshape(-1, 10)
        normalized = flat_dynamics.float().clone()
        normalized[:, :6] = torch.sign(normalized[:, :6]) * torch.log1p(
            normalized[:, :6].abs() * 1000.0
        )
        normalized[:, 6:] = torch.log1p(normalized[:, 6:].clamp_min(0.0))
        types = self.parameter_embedding(
            torch.arange(10, device=flat_dynamics.device)
        ).unsqueeze(0)
        tokens = self.output(self.value_projection(normalized.unsqueeze(-1)) + types)
        if len(object_shape) == 2:
            return tokens.reshape(object_shape[0], object_shape[1] * 10, -1)
        return tokens


class PhyContextConditionEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        context_dim: int = 4096,
        scene_token_count: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.scene_encoder = ScenePointEncoder(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            token_count=scene_token_count,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.physics_encoder = PhysicsControlEncoder(hidden_dim, context_dim)
        self.dynamics_encoder = BaseDynamicsEncoder(hidden_dim, context_dim)
        self.state_encoder = InitialStateEncoder(hidden_dim, context_dim)

    def forward(self, scene, controls, dynamics, initial_state) -> torch.Tensor:
        scene_tokens = self.scene_encoder(scene)
        batch_size = scene_tokens.shape[0]
        if controls.ndim == 2 and batch_size == 1 and controls.shape[0] != 1:
            controls = controls.unsqueeze(0)
        if dynamics.ndim == 2 and batch_size == 1 and dynamics.shape[0] != 1:
            dynamics = dynamics.unsqueeze(0)
        if initial_state.ndim == 2 and batch_size == 1 and initial_state.shape[0] != 1:
            initial_state = initial_state.unsqueeze(0)
        physics_tokens = self.physics_encoder(controls)
        dynamics_tokens = self.dynamics_encoder(dynamics)
        state_tokens = self.state_encoder(initial_state)
        return torch.cat(
            [scene_tokens, physics_tokens, dynamics_tokens, state_tokens], dim=1
        )


def compose_wan_context(
    text_context: list[torch.Tensor] | torch.Tensor,
    condition_tokens: torch.Tensor,
    max_tokens: int = 512,
) -> list[torch.Tensor]:
    """Append scene/physics tokens while preserving Wan's existing context interface."""
    if isinstance(text_context, torch.Tensor):
        if text_context.ndim == 2:
            text_items = [text_context]
        elif text_context.ndim == 3:
            text_items = list(text_context)
        else:
            raise ValueError("text_context must have shape L x C or B x L x C")
    else:
        text_items = text_context
    if len(text_items) != len(condition_tokens):
        raise ValueError("text and condition batch sizes differ")
    condition_count = condition_tokens.shape[1]
    text_budget = max_tokens - condition_count
    if text_budget <= 0:
        raise ValueError("condition tokens leave no room for text context")
    return [
        torch.cat([text[:text_budget], condition_tokens[index]], dim=0)
        for index, text in enumerate(text_items)
    ]
