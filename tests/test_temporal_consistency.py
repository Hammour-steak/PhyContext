import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "phycontext"))

from train_wan_formal import clean_latent_lpips_loss  # noqa: E402
from wan_training import latent_temporal_consistency_loss  # noqa: E402


class FakeVAE:
    def __init__(self):
        self.temporal_lengths = []

    def decode(self, values):
        value = values[0]
        self.temporal_lengths.append(int(value.shape[1]))
        return [value]


class FakePerceptual(torch.nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).abs().mean(dim=(1, 2, 3))


class TemporalConsistencyTest(unittest.TestCase):
    def test_temporal_loss_is_zero_for_matching_clean_latents(self):
        target = torch.randn(4, 5, 3, 3)
        loss = latent_temporal_consistency_loss([target], [target])
        self.assertEqual(float(loss), 0.0)

    def test_temporal_loss_matches_motion_not_static_frames(self):
        target = torch.zeros(4, 5, 3, 3)
        target[:, 1:] = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)
        stationary = torch.zeros_like(target)
        loss = latent_temporal_consistency_loss([stationary], [target])
        self.assertGreater(float(loss), 0.0)

    def test_lpips_decodes_contiguous_temporal_windows(self):
        predicted = torch.randn(3, 7, 8, 8, requires_grad=True)
        target = torch.randn(3, 7, 8, 8)
        vae = FakeVAE()
        loss = clean_latent_lpips_loss(
            [predicted],
            [target],
            vae,
            FakePerceptual(),
            window_count=2,
            resolution=16,
            temporal_window=3,
        )
        loss.backward()
        self.assertEqual(vae.temporal_lengths, [3, 3, 3, 3])
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_lpips_excludes_the_clean_condition_frame(self):
        predicted = torch.zeros(3, 7, 8, 8, requires_grad=True)
        with torch.no_grad():
            predicted[:, 0].fill_(10.0)
        target = torch.zeros_like(predicted)
        loss = clean_latent_lpips_loss(
            [predicted],
            [target],
            FakeVAE(),
            FakePerceptual(),
            window_count=2,
            resolution=16,
            temporal_window=3,
        )
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertEqual(float(predicted.grad[:, 0].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
