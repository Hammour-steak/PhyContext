import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from conditioning_model import PhyContextConditionEncoder
from train_wan_formal import (
    configure_training_stage,
    learning_rate_factor,
    training_condition_mode,
)
from wan_training import (
    DirectConditionModulator,
    LoRALinear,
    balanced_motion_loss_mask,
    inject_direct_condition_modulation,
    inject_cross_attention_lora,
    inject_trajectory_conditioning,
    index_nominal_trajectory_records,
    is_paired_response_epoch,
    load_condition_checkpoint,
    latent_correspondence_motion,
    latent_correspondence_trajectory,
    latent_motion_supervision_losses,
    lora_parameters,
    make_ti2v_flow_batch,
    make_complete_sweep_response_batches,
    make_formal_training_batch,
    masked_flow_loss,
    masked_flow_response_loss,
    recover_clean_latents,
    select_complete_sweep_levels,
    select_round_robin_records,
    select_sweep_endpoint_pairs,
    select_trajectory_input_records,
    set_direct_condition,
    set_trajectory_condition,
    save_condition_checkpoint,
    source_target_motion_envelope,
    structured_direct_condition,
    union_motion_masks,
)


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(8, 8)
        self.k = nn.Linear(8, 8)
        self.v = nn.Linear(8, 8)
        self.o = nn.Linear(8, 8)


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = DummyAttention()

    def forward(
        self, x, e, seq_lens, grid_sizes, freqs, context, context_lens
    ):
        return x + e.mean(dim=2)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 8
        self.blocks = nn.ModuleList([DummyBlock(), DummyBlock()])


class DummyTrajectoryModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.patch_size = (1, 2, 2)
        self.patch_embedding = nn.Conv3d(
            4, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, x, **kwargs):
        return [self.patch_embedding(item.unsqueeze(0)) for item in x]


class WanTrainingTest(unittest.TestCase):
    def test_trajectory_warmup_freezes_every_physics_specific_module(self) -> None:
        model = DummyModel()
        inject_direct_condition_modulation(
            model,
            rank=2,
            alpha=2.0,
            input_dim=12,
        )
        encoder = PhyContextConditionEncoder(
            hidden_dim=16,
            context_dim=32,
            scene_token_count=4,
            num_heads=4,
            num_layers=1,
        )
        configure_training_stage(encoder, model, "trajectory_warmup")
        self.assertEqual(training_condition_mode("trajectory_warmup"), "scene_only")
        self.assertTrue(any(p.requires_grad for p in encoder.scene_encoder.parameters()))
        for module in (
            encoder.physics_encoder,
            encoder.dynamics_encoder,
            encoder.state_encoder,
            model.phycontext_direct_modulator,
        ):
            self.assertFalse(any(p.requires_grad for p in module.parameters()))

        configure_training_stage(encoder, model, "joint")
        self.assertEqual(training_condition_mode("joint"), "full")
        for module in (
            encoder.physics_encoder,
            encoder.dynamics_encoder,
            encoder.state_encoder,
            model.phycontext_direct_modulator,
        ):
            self.assertTrue(all(p.requires_grad for p in module.parameters()))

    def test_nominal_trajectory_is_shared_across_sweep_targets(self) -> None:
        base = {
            "sample_id": "scene__base",
            "base_scene_id": "scene",
            "record": {"sweep": {"mode": "base"}},
        }
        low = {
            "sample_id": "scene__friction_00",
            "base_scene_id": "scene",
            "record": {"sweep": {"mode": "one_factor"}},
        }
        high = {
            "sample_id": "scene__friction_04",
            "base_scene_id": "scene",
            "record": {"sweep": {"mode": "one_factor"}},
        }
        nominal = index_nominal_trajectory_records([base, low, high])
        self.assertEqual(
            select_trajectory_input_records(
                [low, high], nominal, source="nominal_base"
            ),
            [base, base],
        )
        self.assertEqual(
            select_trajectory_input_records([low, high], nominal, source="target"),
            [low, high],
        )

    def test_target_trajectory_stays_bound_to_each_physical_sample(self) -> None:
        low = {
            "sample_id": "scene__friction_00",
            "base_scene_id": "scene",
            "record": {"sweep": {"mode": "one_factor"}},
        }
        high = {
            "sample_id": "scene__friction_04",
            "base_scene_id": "scene",
            "record": {"sweep": {"mode": "one_factor"}},
        }
        nominal = {
            "scene": {
                "sample_id": "scene__base",
                "base_scene_id": "scene",
                "record": {"sweep": {"mode": "base"}},
            }
        }
        self.assertEqual(
            select_trajectory_input_records([low, high], nominal, source="target"),
            [low, high],
        )

    def test_transport_envelope_supervises_source_removal_and_target_occupancy(self) -> None:
        mask = torch.zeros(1, 3, 4, 6, dtype=torch.uint8)
        mask[:, 0, 2, 1] = 1
        mask[:, 1, 2, 3] = 1
        mask[:, 2, 1, 5] = 1
        envelope = source_target_motion_envelope(mask)
        self.assertEqual(int(envelope.sum()), 5)
        self.assertTrue(bool(envelope[0, 1, 2, 1]))
        self.assertTrue(bool(envelope[0, 1, 2, 3]))
        self.assertTrue(bool(envelope[0, 2, 2, 1]))
        self.assertTrue(bool(envelope[0, 2, 1, 5]))

    def test_trajectory_patch_conditioning_starts_as_exact_identity(self) -> None:
        model = DummyTrajectoryModel()
        videos = [torch.randn(4, 3, 6, 8), torch.randn(4, 3, 6, 8)]
        expected = [item.detach().clone() for item in model(videos)]
        conditioner = inject_trajectory_conditioning(model, rank=3)
        point_maps = [torch.randn(18, 3, 6, 8) for _ in videos]
        self.assertTrue(
            set_trajectory_condition(model, point_maps)
        )
        actual = model(videos)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference)
        sum(item.sum() for item in actual).backward()
        self.assertIsNotNone(conditioner.output_projection.weight.grad)
        self.assertGreater(
            float(conditioner.output_projection.weight.grad.abs().sum()), 0.0
        )

    def test_trajectory_conditioning_round_trip(self) -> None:
        source_model = DummyTrajectoryModel()
        inject_cross_attention_lora(source_model, rank=2, alpha=2)
        source_trajectory = inject_trajectory_conditioning(source_model, rank=3)
        with torch.no_grad():
            source_trajectory.output_projection.weight.fill_(0.125)
        source_condition = nn.Linear(3, 4)
        metadata = {
            "schema": "phycontext.wan_condition_adapter.v2",
            "lora": {"rank": 2, "alpha": 2.0, "module_count": 8},
            "trajectory_conditioning": {
                "enabled": True,
                "rank": 3,
                "representation": "dense_point_tracks",
                "input_channels": 18,
                "architecture": "framewise_patch",
                "patch_size": [1, 2, 2],
            },
        }
        with TemporaryDirectory() as directory:
            adapter = Path(directory)
            save_condition_checkpoint(
                adapter, source_model, source_condition, metadata
            )
            target_model = DummyTrajectoryModel()
            target_condition = nn.Linear(3, 4)
            load_condition_checkpoint(adapter, target_model, target_condition)
        torch.testing.assert_close(
            target_model.phycontext_trajectory_conditioner.output_projection.weight,
            source_trajectory.output_projection.weight,
        )

    def test_formal_learning_rate_warms_up_and_cosine_decays(self) -> None:
        factors = [
            learning_rate_factor(step, 100, warmup_ratio=0.1, minimum_ratio=0.1)
            for step in range(100)
        ]
        self.assertAlmostEqual(factors[0], 0.1)
        self.assertAlmostEqual(factors[9], 1.0)
        self.assertGreater(factors[10], factors[50])
        self.assertGreater(factors[50], factors[-1])
        self.assertGreaterEqual(factors[-1], 0.1)

    def test_adapter_round_trip_restores_lora_and_condition_encoder(self) -> None:
        source_model = DummyModel()
        inject_cross_attention_lora(source_model, rank=2, alpha=2)
        source_condition = nn.Linear(3, 4)
        with torch.no_grad():
            source_model.blocks[0].cross_attn.q.up.weight.fill_(0.25)
            source_condition.weight.fill_(0.5)
        metadata = {
            "schema": "phycontext.wan_condition_adapter.v2",
            "lora": {"rank": 2, "alpha": 2.0, "module_count": 8},
        }
        with TemporaryDirectory() as directory:
            adapter = Path(directory)
            save_condition_checkpoint(
                adapter, source_model, source_condition, metadata
            )
            target_model = DummyModel()
            target_condition = nn.Linear(3, 4)
            loaded = load_condition_checkpoint(
                adapter, target_model, target_condition
            )
        self.assertEqual(loaded, metadata)
        torch.testing.assert_close(
            target_model.blocks[0].cross_attn.q.up.weight,
            source_model.blocks[0].cross_attn.q.up.weight,
        )
        torch.testing.assert_close(
            target_condition.weight, source_condition.weight
        )

    def test_training_records_rotate_across_base_scenes(self) -> None:
        records = [
            {"sample_id": f"{scene}-{index}", "base_scene_id": scene}
            for scene in ("a", "b", "c")
            for index in range(3)
        ]
        selected = select_round_robin_records(records, 7, 3)
        self.assertEqual(
            [record["sample_id"] for record in selected],
            ["a-0", "b-0", "c-0", "a-1", "b-1", "c-1", "a-2"],
        )

    def test_direct_modulation_starts_as_exact_identity_delta(self) -> None:
        model = DummyModel()
        x = torch.randn(1, 3, 8)
        e = torch.randn(1, 3, 6, 8)
        context = torch.randn(1, 10, 8)
        arguments = (None, None, None, context, None)
        expected = model.blocks[0](x, e, *arguments).detach()
        modulator = inject_direct_condition_modulation(
            model, rank=2, alpha=2, input_dim=12
        )
        set_direct_condition(
            model,
            torch.tensor([[1.0, 0.2, 0.1]]),
            torch.zeros(1, 9),
        )
        actual = model.blocks[0](x, e, *arguments)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertIsInstance(modulator, DirectConditionModulator)
        self.assertIsNotNone(modulator.up[0].weight.grad)
        self.assertIsNotNone(modulator.down.weight.grad)

    def test_disabled_direct_modulation_is_an_exact_zero_delta(self) -> None:
        model = DummyModel()
        x = torch.randn(1, 3, 8)
        e = torch.randn(1, 3, 6, 8)
        context = torch.randn(1, 10, 8)
        arguments = (None, None, None, context, None)
        expected = model.blocks[0](x, e, *arguments).detach()
        modulator = inject_direct_condition_modulation(
            model, rank=2, alpha=2, input_dim=12
        )
        with torch.no_grad():
            modulator.down.bias.fill_(1.0)
            modulator.up[0].weight.fill_(1.0)
        set_direct_condition(
            model,
            torch.tensor([[1.0, 0.2, 0.1]]),
            torch.zeros(1, 9),
            enabled=False,
        )
        actual = model.blocks[0](x, e, *arguments)
        torch.testing.assert_close(actual, expected)

    def test_direct_modulation_uses_current_normalization(self) -> None:
        controls = torch.tensor([[1.0, 0.2, 0.1]])
        state = torch.zeros(1, 9)
        model = DummyModel()
        modulator = inject_direct_condition_modulation(
            model, rank=2, alpha=2, input_dim=12
        )
        set_direct_condition(model, controls, state)
        self.assertAlmostEqual(
            modulator._structured_condition[0, 1].item(),
            torch.log(controls[0, 1]).item(),
        )
        self.assertAlmostEqual(modulator._structured_condition[0, 2].item(), -0.8)

    def test_direct_modulation_round_trip(self) -> None:
        source_model = DummyModel()
        inject_cross_attention_lora(source_model, rank=2, alpha=2)
        source_modulator = inject_direct_condition_modulation(
            source_model, rank=2, alpha=2, input_dim=12
        )
        source_condition = nn.Linear(3, 4)
        with torch.no_grad():
            for head in source_modulator.up:
                head.weight.fill_(0.125)
        metadata = {
            "schema": "phycontext.wan_condition_adapter.v2",
            "lora": {"rank": 2, "alpha": 2.0, "module_count": 8},
            "direct_modulation": {
                "enabled": True,
                "source": "structured_controls_v2",
                "architecture": "per_layer_v2",
                "rank": 2,
                "alpha": 2.0,
                "input_dim": 12,
            },
        }
        with TemporaryDirectory() as directory:
            adapter = Path(directory)
            save_condition_checkpoint(
                adapter, source_model, source_condition, metadata
            )
            target_model = DummyModel()
            target_condition = nn.Linear(3, 4)
            load_condition_checkpoint(adapter, target_model, target_condition)
        torch.testing.assert_close(
            target_model.phycontext_direct_modulator.up[0].weight,
            source_modulator.up[0].weight,
        )
    def test_sweep_endpoint_pairs_restore_canonical_base_level(self) -> None:
        axes = ("mass_kg", "contact_friction", "contact_restitution")
        base = {
            "sample_id": "scene-base",
            "base_scene_id": "scene",
            "record": {
                "sweep": {
                    "mode": "base",
                    "axis": None,
                    "level_count": 5,
                    "level_index": 2,
                    "base_level_indices": {axis: 2 for axis in axes},
                }
            },
        }
        records = [base]
        for axis in axes:
            for level in (0, 1, 3, 4):
                records.append(
                    {
                        "sample_id": f"scene-{axis}-{level}",
                        "base_scene_id": "scene",
                        "record": {
                            "sweep": {
                                "mode": "sweep",
                                "axis": axis,
                                "level_count": 5,
                                "level_index": level,
                            }
                        },
                    }
                )
        pairs = select_sweep_endpoint_pairs(records, 1)
        self.assertEqual([pair["axis"] for pair in pairs], list(axes))
        self.assertTrue(all(pair["low"]["sample_id"].endswith("-0") for pair in pairs))
        self.assertTrue(all(pair["high"]["sample_id"].endswith("-4") for pair in pairs))

    def test_complete_sweep_levels_include_canonical_base_in_order(self) -> None:
        axes = ("mass_kg", "contact_friction", "contact_restitution")
        base = {
            "sample_id": "scene-base",
            "base_scene_id": "scene",
            "record": {
                "sweep": {
                    "mode": "base",
                    "axis": None,
                    "level_count": 5,
                    "level_index": 2,
                    "base_level_indices": {axis: 2 for axis in axes},
                }
            },
        }
        records = [base]
        for level in (0, 1, 3, 4):
            records.append(
                {
                    "sample_id": f"scene-contact_friction-{level}",
                    "base_scene_id": "scene",
                    "record": {
                        "sweep": {
                            "mode": "sweep",
                            "axis": "contact_friction",
                            "level_count": 5,
                            "level_index": level,
                        }
                    },
                }
            )
        levels = select_complete_sweep_levels(
            records, "scene", "contact_friction"
        )
        self.assertEqual(
            [item["sample_id"] for item in levels],
            [
                "scene-contact_friction-0",
                "scene-contact_friction-1",
                "scene-base",
                "scene-contact_friction-3",
                "scene-contact_friction-4",
            ],
        )

    def test_complete_sweep_response_batches_cover_local_and_global_changes(self) -> None:
        levels = [{"sample_id": f"level-{index}"} for index in range(5)]
        batches = make_complete_sweep_response_batches(levels)
        self.assertEqual(
            [[item["sample_id"] for item in batch] for batch in batches],
            [
                ["level-0", "level-1"],
                ["level-1", "level-2"],
                ["level-2", "level-3"],
                ["level-3", "level-4"],
                ["level-0", "level-4"],
            ],
        )

    def test_response_schedule_alternates_complete_pair_epochs(self) -> None:
        flags = [is_paired_response_epoch(step, 3, 2) for step in range(12)]
        self.assertEqual(
            flags,
            [True, True, True, False, False, False] * 2,
        )

    def test_lora_starts_as_exact_identity_delta(self) -> None:
        model = DummyModel()
        inputs = torch.randn(3, 8)
        expected = model.blocks[0].cross_attn.q(inputs).detach()
        names = inject_cross_attention_lora(model, rank=2, alpha=2)
        self.assertEqual(len(names), 8)
        self.assertIsInstance(model.blocks[0].cross_attn.q, LoRALinear)
        actual = model.blocks[0].cross_attn.q(inputs)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        active = model.blocks[0].cross_attn.q
        self.assertTrue(all(parameter.grad is not None for parameter in active.down.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in active.up.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in active.base.parameters()))
        self.assertEqual(len(list(lora_parameters(model))), 16)

    def test_formal_schedule_is_sixty_forty_and_axis_weighted(self) -> None:
        records = [
            {"sample_id": f"record-{index}", "base_scene_id": f"base-{index}"}
            for index in range(20)
        ]
        pairs = [
            {
                "axis": axis,
                "low": {"sample_id": f"{axis}-low-{index}"},
                "high": {"sample_id": f"{axis}-high-{index}"},
            }
            for axis in ("mass_kg", "contact_friction", "contact_restitution")
            for index in range(4)
        ]
        modes = []
        axes = []
        rank_batches = []
        for step in range(25):
            batch, mode, axis = make_formal_training_batch(
                records,
                pairs,
                step=step,
                accumulation_index=0,
                gradient_accumulation=1,
                rank=0,
                world_size=2,
                seed=13,
            )
            modes.append(mode)
            if axis is not None:
                axes.append(axis)
            rank_batches.append([item["sample_id"] for item in batch])
        self.assertEqual(modes.count("response"), 10)
        self.assertEqual(modes.count("ordinary"), 15)
        self.assertEqual(axes.count("contact_friction"), 4)
        self.assertEqual(axes.count("contact_restitution"), 4)
        self.assertEqual(axes.count("mass_kg"), 2)

        rank_one = [
            item["sample_id"]
            for item in make_formal_training_batch(
                records,
                pairs,
                step=2,
                accumulation_index=0,
                gradient_accumulation=1,
                rank=1,
                world_size=2,
                seed=13,
            )[0]
        ]
        self.assertNotEqual(rank_batches[2], rank_one)

    def test_ti2v_flow_keeps_first_latent_frame_clean(self) -> None:
        latent = torch.randn(4, 5, 6, 8)
        batch = make_ti2v_flow_batch([latent], torch.tensor([0.7]))
        torch.testing.assert_close(batch["noisy_latents"][0][:, 0], latent[:, 0])
        self.assertTrue((batch["timesteps"][0][: 3 * 4] == 0).all())
        self.assertEqual(batch["seq_len"], 5 * 3 * 4)
        self.assertEqual(float(batch["loss_masks"][0][:, 0].sum()), 0.0)

    def test_shared_noise_isolates_endpoint_latent_difference(self) -> None:
        low = torch.zeros(2, 3, 4, 4)
        high = torch.ones_like(low)
        batch = make_ti2v_flow_batch(
            [low, high],
            torch.tensor([0.7, 0.7]),
            shared_noise=True,
            generator=torch.Generator().manual_seed(7),
        )
        torch.testing.assert_close(
            batch["targets"][1] - batch["targets"][0], -torch.ones_like(low)
        )

    def test_clean_latent_recovery_inverts_rectified_flow(self) -> None:
        latent = torch.randn(3, 4, 2, 2)
        batch = make_ti2v_flow_batch(
            [latent],
            torch.tensor([0.65]),
            generator=torch.Generator().manual_seed(9),
        )
        recovered = recover_clean_latents(
            batch["noisy_latents"], batch["targets"], torch.tensor([0.65])
        )[0]
        torch.testing.assert_close(recovered[:, 1:], latent[:, 1:])

    def test_latent_trajectory_matches_moving_object_and_has_gradients(self) -> None:
        target = torch.zeros(3, 4, 5, 6)
        motion = torch.zeros(1, 4, 5, 6, dtype=torch.uint8)
        positions = [(1, 1), (1, 2), (2, 3), (3, 4)]
        for frame, (y, x) in enumerate(positions):
            target[:, frame, y, x] = torch.tensor([3.0, -2.0, 1.0])
            motion[:, frame, y, x] = 1
        exact = target.clone().requires_grad_(True)
        predicted_centers, target_centers, valid = latent_correspondence_trajectory(
            exact, target, motion, temperature=0.03
        )
        torch.testing.assert_close(
            predicted_centers[valid], target_centers[valid], atol=1e-4, rtol=1e-4
        )
        exact_loss = latent_motion_supervision_losses(
            [exact], [target], [motion], temperature=0.03
        )["center"]
        exact_loss.backward()
        self.assertIsNotNone(exact.grad)

        shifted = torch.roll(target, shifts=1, dims=-1).requires_grad_(True)
        shifted_loss = latent_motion_supervision_losses(
            [shifted], [target], [motion], temperature=0.03
        )["center"]
        self.assertGreater(
            float(shifted_loss.detach()), float(exact_loss.detach()) + 0.01
        )

    def test_distribution_rejects_symmetric_duplicates_with_correct_center(self) -> None:
        target = torch.zeros(3, 4, 5, 7)
        motion = torch.zeros(1, 4, 5, 7, dtype=torch.uint8)
        feature = torch.tensor([3.0, -2.0, 1.0])
        for frame in range(4):
            target[:, frame, 2, 3] = feature
            motion[:, frame, 2, 3] = 1

        duplicate = torch.zeros_like(target)
        duplicate[:, :, 2, 1] = feature[:, None]
        duplicate[:, :, 2, 5] = feature[:, None]
        duplicate.requires_grad_(True)
        predicted, expected, _, _, valid = latent_correspondence_motion(
            duplicate,
            target,
            motion,
            temperature=0.03,
        )
        torch.testing.assert_close(
            predicted[valid], expected[valid], atol=1e-4, rtol=1e-4
        )

        exact_losses = latent_motion_supervision_losses(
            [target], [target], [motion], temperature=0.03
        )
        duplicate_losses = latent_motion_supervision_losses(
            [duplicate], [target], [motion], temperature=0.03
        )
        self.assertLess(float(duplicate_losses["center"].detach()), 1e-4)
        self.assertGreater(
            float(duplicate_losses["distribution"].detach()),
            float(exact_losses["distribution"].detach()) + 0.1,
        )
        duplicate_losses["distribution"].backward()
        self.assertIsNotNone(duplicate.grad)

    def test_velocity_loss_rejects_wrong_frame_to_frame_displacement(self) -> None:
        target = torch.zeros(3, 5, 5, 7)
        motion = torch.zeros(1, 5, 5, 7, dtype=torch.uint8)
        feature = torch.tensor([3.0, -2.0, 1.0])
        target_positions = [1, 1, 2, 3, 4]
        predicted_positions = [1, 1, 1, 3, 4]
        wrong = torch.zeros_like(target)
        for frame, (target_x, predicted_x) in enumerate(
            zip(target_positions, predicted_positions)
        ):
            target[:, frame, 2, target_x] = feature
            wrong[:, frame, 2, predicted_x] = feature
            motion[:, frame, 2, target_x] = 1

        exact_losses = latent_motion_supervision_losses(
            [target], [target], [motion], temperature=0.03
        )
        wrong_losses = latent_motion_supervision_losses(
            [wrong], [target], [motion], temperature=0.03
        )
        self.assertGreater(
            float(wrong_losses["velocity"].detach()),
            float(exact_losses["velocity"].detach()) + 0.01,
        )

    def test_masked_loss_ignores_condition_frame(self) -> None:
        target = torch.zeros(2, 3, 2, 2)
        prediction = torch.zeros_like(target)
        prediction[:, 0] = 100
        prediction[:, 1:] = 2
        mask = torch.ones_like(target)
        mask[:, 0] = 0
        loss = masked_flow_loss([prediction], [target], [mask])
        self.assertAlmostEqual(float(loss), 4.0)

    def test_balanced_motion_mask_reserves_foreground_loss_share(self) -> None:
        base = torch.ones(2, 3, 2, 2)
        motion = torch.zeros(1, 3, 2, 2)
        motion[:, :, 0, 0] = 1
        weighted = balanced_motion_loss_mask(base, motion, 0.75)
        expanded = motion.expand_as(base).bool()
        foreground_share = weighted[expanded].sum() / weighted.sum()
        self.assertAlmostEqual(float(foreground_share), 0.75, places=6)
        self.assertAlmostEqual(float(weighted.sum()), float(base.sum()), places=6)

    def test_sweep_motion_union_preserves_per_frame_trajectory_envelope(self) -> None:
        low = torch.zeros(1, 3, 2, 3, dtype=torch.uint8)
        high = torch.zeros_like(low)
        low[:, :, 0, 0] = 1
        high[:, :, 1, 2] = 1
        envelope = union_motion_masks([low, high])
        self.assertEqual(envelope.dtype, torch.uint8)
        self.assertEqual(int(envelope.sum()), 6)
        self.assertTrue(torch.equal(envelope[:, :, 0, 0], torch.ones(1, 3)))
        self.assertTrue(torch.equal(envelope[:, :, 1, 2], torch.ones(1, 3)))

    def test_response_loss_matches_low_high_target_delta(self) -> None:
        low = torch.zeros(2, 3, 2, 2)
        high = torch.ones_like(low)
        masks = [torch.ones_like(low), torch.ones_like(high)]
        exact = masked_flow_response_loss([low, high], [low, high], masks)
        collapsed = masked_flow_response_loss([low, low], [low, high], masks)
        self.assertEqual(float(exact), 0.0)
        self.assertAlmostEqual(float(collapsed), 1.0)

    def test_response_loss_accepts_explicit_motion_region(self) -> None:
        low = torch.zeros(1, 2, 2, 2)
        high_target = torch.ones_like(low)
        high_prediction = torch.zeros_like(low)
        high_prediction[:, :, 0, 0] = 1
        masks = [torch.ones_like(low), torch.ones_like(low)]
        response_mask = torch.zeros_like(low)
        response_mask[:, :, 0, 0] = 1
        loss = masked_flow_response_loss(
            [low, high_prediction],
            [low, high_target],
            masks,
            response_mask=response_mask,
        )
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
