import math
import unittest

import torch

from common import (
    GuidanceContext,
    GuidanceConfig,
    GuidanceState,
    apply_guidance,
    apply_model_guidance,
    epsilon_to_x0,
    flow_velocity_to_x0,
    guidance_domain,
    x0_to_epsilon,
    x0_to_flow_velocity,
)
from common.spectral import decompose_residual, frequency_masks


class ParameterizationTests(unittest.TestCase):
    def test_flow_roundtrip(self):
        torch.manual_seed(0)
        z = torch.randn(2, 4, 8, 8)
        velocity = torch.randn_like(z)
        t = torch.tensor([0.2, 0.8])
        x0 = flow_velocity_to_x0(z, velocity, t)
        recovered = x0_to_flow_velocity(z, x0, t)
        torch.testing.assert_close(recovered, velocity)

    def test_epsilon_roundtrip(self):
        torch.manual_seed(1)
        x = torch.randn(2, 4, 8, 8)
        epsilon = torch.randn_like(x)
        alpha = torch.tensor([0.8, 0.3])
        sigma = torch.sqrt(1 - alpha.square())
        x0 = epsilon_to_x0(x, epsilon, alpha, sigma)
        recovered = x0_to_epsilon(x, x0, alpha, sigma)
        torch.testing.assert_close(recovered, epsilon)


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.cond = torch.randn(2, 4, 16, 16)
        self.uncond = torch.randn_like(self.cond)

    def test_cfg_definition(self):
        config = GuidanceConfig(method="cfg", scale=4)
        actual = apply_guidance(self.cond, self.uncond, config)
        expected = self.uncond + 4 * (self.cond - self.uncond)
        torch.testing.assert_close(actual, expected)

    def test_reference_execution_domains(self):
        expected = {
            "cfg": "native",
            "cfg_rescale": "native",
            "c2fg": "native",
            "apg": "clean",
            "fdg": "clean",
            "rptg": "clean",
            "tfdg": "clean",
        }
        self.assertEqual(
            {method: guidance_domain(method) for method in expected},
            expected,
        )

    def test_apg_algorithm_one_order(self):
        config = GuidanceConfig(
            method="apg",
            scale=4,
            momentum=-0.75,
            norm_threshold=2.5,
            parallel_weight=0,
        )
        state = GuidanceState(momentum_buffer=torch.ones_like(self.cond) * 0.1)
        actual = apply_guidance(self.cond, self.uncond, config, state)

        running = self.cond - self.uncond - 0.75 * 0.1
        norm = torch.linalg.vector_norm(
            running.float(), dim=(1, 2, 3), keepdim=True
        )
        clipped = running * torch.clamp(2.5 / norm, max=1)
        unit = torch.nn.functional.normalize(self.cond.double(), dim=(1, 2, 3))
        parallel = (clipped.double() * unit).sum(
            dim=(1, 2, 3), keepdim=True
        ) * unit
        orthogonal = clipped - parallel.to(clipped.dtype)
        expected = self.cond + 3 * orthogonal
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(state.momentum_buffer, running)

    def test_fdg_equal_scales_equals_cfg(self):
        config = GuidanceConfig(
            method="fdg",
            scale=3,
            fdg_low_scale=3,
        )
        fdg = apply_guidance(self.cond, self.uncond, config)
        cfg = apply_guidance(
            self.cond, self.uncond, GuidanceConfig(method="cfg", scale=3)
        )
        torch.testing.assert_close(fdg, cfg, atol=2e-5, rtol=2e-5)

    def test_rptg_scale_one_is_conditional(self):
        config = GuidanceConfig(method="rptg", scale=1)
        actual = apply_guidance(self.cond, self.uncond, config)
        torch.testing.assert_close(actual, self.cond)

    def test_tfdg_all_weights_equal_cfg(self):
        context = GuidanceContext(alpha=0.7, sigma=0.7, noise_time=0.5)
        config = GuidanceConfig(
            method="tfdg", scale=4, low_min_cfg=4, high_min_cfg=4,
            low_eta=0, high_eta=0,
        )
        actual = apply_guidance(self.cond, self.uncond, config, context=context)
        expected = apply_guidance(self.cond, self.uncond, GuidanceConfig(method="cfg", scale=4))
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_anchor_cfg_matches_native_cfg_for_flow_velocity(self):
        t = torch.tensor([0.2, 0.8])
        z = torch.randn_like(self.cond)
        v_cond = self.cond
        v_uncond = self.uncond
        scale = 4.0
        native = v_uncond + scale * (v_cond - v_uncond)
        anchor = apply_model_guidance(
            v_cond,
            v_uncond,
            GuidanceConfig(method="tfdg", scale=scale, low_min_cfg=scale,
                           high_min_cfg=scale, low_eta=0, high_eta=0),
            to_clean=lambda value: flow_velocity_to_x0(z, value, t),
            from_clean=lambda value: x0_to_flow_velocity(z, value, t),
            context=GuidanceContext(alpha=0.5, sigma=0.5),
        )
        torch.testing.assert_close(anchor, native, atol=2e-5, rtol=2e-5)

    def test_anchor_cfg_matches_native_cfg_for_epsilon(self):
        x = torch.randn_like(self.cond)
        alpha = torch.tensor([0.8, 0.4])
        sigma = torch.sqrt(1 - alpha.square())
        scale = 4.0
        native = self.uncond + scale * (self.cond - self.uncond)
        anchor = apply_model_guidance(
            self.cond,
            self.uncond,
            GuidanceConfig(method="tfdg", scale=scale, low_min_cfg=scale,
                           high_min_cfg=scale, low_eta=0, high_eta=0),
            to_clean=lambda value: epsilon_to_x0(x, value, alpha, sigma),
            from_clean=lambda value: x0_to_epsilon(x, value, alpha, sigma),
            context=GuidanceContext(alpha=0.5, sigma=0.5),
        )
        torch.testing.assert_close(anchor, native, atol=2e-5, rtol=2e-5)

    def test_frequency_masks_are_complementary(self):
        low, high = frequency_masks(self.cond, cutoff=0.3)
        torch.testing.assert_close(low + high, torch.ones_like(low))

    def test_four_components_reconstruct_residual(self):
        _, parts = decompose_residual(self.cond, self.uncond, cutoff=0.3)
        spectrum = sum(parts.values())
        reconstructed = torch.fft.irfft2(
            spectrum, s=self.cond.shape[-2:], norm="ortho"
        )
        torch.testing.assert_close(
            reconstructed, self.cond - self.uncond, atol=2e-5, rtol=2e-5
        )

    def test_tfdg_noisy_endpoint_is_finite(self):
        config = GuidanceConfig(
            method="tfdg", scale=15, low_min_cfg=1, high_min_cfg=3,
            low_eta=0.5, high_eta=1,
        )
        actual = apply_guidance(
            self.cond,
            self.uncond,
            config,
            context=GuidanceContext(alpha=0, sigma=1, noise_time=1),
        )
        self.assertTrue(torch.isfinite(actual).all())

    def test_tfdg_partial_recovery_uses_explicit_ceilings(self):
        config = GuidanceConfig(
            method="tfdg", scale=5, low_min_cfg=1, low_max_cfg=1.1,
            high_min_cfg=1, high_max_cfg=1.25, low_eta=4, high_eta=0.5,
        )
        # At the clean endpoint g=1, each branch must recover only to its
        # selected ceiling rather than the global CFG scale.
        clean = apply_guidance(
            self.cond, self.uncond, config,
            context=GuidanceContext(alpha=1, sigma=0, noise_time=0),
        )
        full = apply_guidance(
            self.cond, self.uncond,
            GuidanceConfig(method="cfg", scale=5),
        )
        self.assertFalse(torch.allclose(clean, full))
        self.assertTrue(torch.isfinite(clean).all())

    def test_c2fg_recovers_cfg_at_clean_endpoint(self):
        config = GuidanceConfig(method="c2fg", scale=4, c2fg_lambda=1)
        actual = apply_guidance(
            self.cond, self.uncond, config,
            context=GuidanceContext(noise_time=0),
        )
        expected = apply_guidance(self.cond, self.uncond, GuidanceConfig(method="cfg", scale=4))
        torch.testing.assert_close(actual, expected)

    def test_c2fg_matches_reference_exponential_schedule(self):
        config = GuidanceConfig(method="c2fg", scale=4, c2fg_lambda=1)
        actual = apply_guidance(
            self.cond,
            self.uncond,
            config,
            context=GuidanceContext(noise_time=1),
        )
        effective = 4 * math.exp(-1)
        expected = self.uncond + effective * (self.cond - self.uncond)
        torch.testing.assert_close(actual, expected)

    def test_cfg_rescale_factor_zero_is_cfg(self):
        config = GuidanceConfig(method="cfg_rescale", scale=4, rescale_factor=0)
        actual = apply_guidance(self.cond, self.uncond, config)
        expected = apply_guidance(self.cond, self.uncond, GuidanceConfig(method="cfg", scale=4))
        torch.testing.assert_close(actual, expected)

    def test_native_methods_do_not_evaluate_anchor_conversion(self):
        def forbidden(_prediction):
            raise AssertionError("native-domain guidance must not convert to x0")

        for method in ("cfg", "cfg_rescale", "c2fg"):
            with self.subTest(method=method):
                config = GuidanceConfig(method=method, scale=4)
                actual = apply_model_guidance(
                    self.cond,
                    self.uncond,
                    config,
                    to_clean=forbidden,
                    from_clean=forbidden,
                    context=GuidanceContext(noise_time=0.5),
                )
                expected = apply_guidance(
                    self.cond,
                    self.uncond,
                    config,
                    context=GuidanceContext(noise_time=0.5),
                )
                torch.testing.assert_close(actual, expected)

    def test_anchor_method_uses_anchor_conversion(self):
        calls = []

        def to_clean(prediction):
            calls.append("to_clean")
            return prediction + 2

        def from_clean(prediction):
            calls.append("from_clean")
            return prediction - 2

        config = GuidanceConfig(method="rptg", scale=4)
        actual = apply_model_guidance(
            self.cond,
            self.uncond,
            config,
            to_clean=to_clean,
            from_clean=from_clean,
        )
        clean_expected = apply_guidance(self.cond + 2, self.uncond + 2, config)
        torch.testing.assert_close(actual, clean_expected - 2)
        self.assertEqual(calls, ["to_clean", "to_clean", "from_clean"])


if __name__ == "__main__":
    unittest.main()
