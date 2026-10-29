import sys
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/VideoPainter/diffusers/src"))
from shotcraft.inference_memory import disable_single_window_feature_cache


class InferenceMemoryTests(unittest.TestCase):
    def test_real_small_transformer_noise_is_identical(self):
        from diffusers import CogVideoXTransformer3DModel

        torch.manual_seed(42)
        model = CogVideoXTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            out_channels=2,
            num_layers=2,
            time_embed_dim=16,
            text_embed_dim=16,
            sample_width=4,
            sample_height=4,
            sample_frames=9,
            patch_size=2,
            max_text_seq_length=3,
            use_rotary_positional_embeddings=False,
        ).eval()
        inputs = dict(
            hidden_states=torch.randn(2, 3, 4, 4, 4),
            encoder_hidden_states=torch.randn(2, 3, 16),
            timestep=torch.ones(2),
            branch_block_masks=torch.zeros(2, 3, 1, 4, 4),
            return_hidden_states=True,
            return_resample_mask=True,
            return_dict=False,
        )
        with torch.inference_mode():
            normal = model(**inputs)
            self.assertEqual(len(normal[1]), 2)
            with disable_single_window_feature_cache(model):
                bounded = model(**inputs)
            torch.testing.assert_close(normal[0], bounded[0], rtol=0, atol=0)
            self.assertEqual(bounded[1], [])
            self.assertIsNone(bounded[2])
            self.assertEqual(len(model(**inputs)[1]), 2)

    def test_hooks_removed_after_error_and_id_path_rejected(self):
        class Toy(torch.nn.Module):
            def forward(self, **kwargs):
                raise RuntimeError("deliberate")

        model = Toy()
        with self.assertRaisesRegex(ValueError, "cross-window"):
            with disable_single_window_feature_cache(model):
                model(return_dict=False, id_pool_resample_learnable=True)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with disable_single_window_feature_cache(model):
                model(return_dict=False)
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model._forward_hooks), 0)
