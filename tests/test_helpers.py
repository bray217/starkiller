"""
Tests for starkiller.helpers pure functions and extinction fitting.

Requirements: pysynphot and extinction must be importable (both are compiled
against NumPy 1.x; use NumPy < 2 or rebuild those packages for NumPy 2.x).
"""
import numpy as np
import pytest

# Gracefully skip the entire module when C-extension dependencies are
# incompatible with the installed NumPy (e.g. NumPy 2.x vs 1.x builds).
pytest.importorskip("extinction", reason="extinction C extension requires NumPy < 2", exc_type=ImportError)
pytest.importorskip("pysynphot", reason="pysynphot requires NumPy < 2", exc_type=ImportError)

import pysynphot as S
from extinction import apply, fitzpatrick99

from starkiller.helpers import (
    downsample_spec,
    fit_extinction,
    lam_vac2air,
    min_dist,
    transform_coords,
)


def _blackbody(wave, T=6000):
    h, c, k = 6.626e-34, 3e8, 1.38e-23
    wave_m = wave * 1e-10
    flux = (2 * h * c**2 / wave_m**5) / (np.exp(h * c / (wave_m * k * T)) - 1)
    return flux / np.nanmedian(flux)


# ── min_dist ────────────────────────────────────────────────────────────────────

class TestMinDist:
    def test_identical_points(self):
        x = np.array([0.0, 1.0])
        assert np.allclose(min_dist(x, x, x, x), 0.0)

    def test_known_345_distance(self):
        x1, y1 = np.array([0.0]), np.array([0.0])
        x2, y2 = np.array([3.0]), np.array([4.0])
        assert np.isclose(min_dist(x1, y1, x2, y2)[0], 5.0)

    def test_returns_minimum_not_mean(self):
        x1, y1 = np.array([0.0]), np.array([0.0])
        x2, y2 = np.array([1.0, 10.0]), np.array([0.0, 0.0])
        assert min_dist(x1, y1, x2, y2)[0] == pytest.approx(1.0)

    def test_output_length_matches_group1(self):
        x1, y1 = np.array([0.0, 5.0, 10.0]), np.zeros(3)
        x2, y2 = np.array([1.0, 2.0]), np.zeros(2)
        assert len(min_dist(x1, y1, x2, y2)) == 3


# ── transform_coords ────────────────────────────────────────────────────────────

class TestTransformCoords:
    _image = np.zeros((100, 100))

    def test_zero_transform_is_identity(self):
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([15.0, 25.0, 35.0])
        xx, yy = transform_coords(x, y, [0, 0, 0], self._image)
        assert np.allclose(xx, x)
        assert np.allclose(yy, y)

    def test_pure_shift(self):
        x, y = np.array([10.0]), np.array([10.0])
        xx, yy = transform_coords(x, y, [5, 3, 0], self._image)
        assert np.isclose(xx[0], 15.0)
        assert np.isclose(yy[0], 13.0)

    def test_quarter_turn_about_center(self):
        cx, cy = 50.0, 50.0
        x, y = np.array([cx + 10.0]), np.array([cy])
        xx, yy = transform_coords(x, y, [0, 0, np.pi / 2], self._image)
        # 90° CCW: (cx+10, cy) → (cx, cy+10)
        assert np.isclose(xx[0], cx, atol=1e-10)
        assert np.isclose(yy[0], cy + 10.0, atol=1e-10)

    def test_full_turn_returns_to_start(self):
        x, y = np.array([23.0]), np.array([47.0])
        xx, yy = transform_coords(x, y, [0, 0, 2 * np.pi], self._image)
        assert np.isclose(xx[0], x[0], atol=1e-10)
        assert np.isclose(yy[0], y[0], atol=1e-10)


# ── lam_vac2air ─────────────────────────────────────────────────────────────────

class TestLamVac2Air:
    def test_air_shorter_than_vacuum(self):
        wave = np.array([4000.0, 5000.0, 6000.0, 7000.0])
        assert np.all(lam_vac2air(wave) < wave)

    def test_formula_consistency(self):
        wave = np.array([5000.0, 6562.8])
        expected = wave / (1 + 2.735182e-4 + 131.4182 / wave**2 + 2.76249e8 / wave**4)
        assert np.allclose(lam_vac2air(wave), expected)

    def test_monotone_preserved(self):
        wave = np.linspace(4000, 9000, 50)
        assert np.all(np.diff(lam_vac2air(wave)) > 0)


# ── fit_extinction ───────────────────────────────────────────────────────────────

class TestFitExtinction:
    """Regression: apply a known E(B-V) via Fitzpatrick99, verify recovery to < 0.05 mag."""

    def _reddened_pair(self, true_ebv, T=6000, Rv=3.1):
        wave = np.arange(4000.0, 8000.0, 2.0)
        flux = _blackbody(wave, T)
        red_flux = apply(fitzpatrick99(wave.astype("double"), true_ebv * Rv, Rv), flux.copy())
        model = S.ArraySpectrum(wave, flux, fluxunits="flam")
        obs = S.ArraySpectrum(wave, red_flux, fluxunits="flam")
        return model, obs

    @pytest.mark.parametrize("true_ebv", [0.1, 0.3, 0.6, 1.0])
    def test_ebv_recovery(self, true_ebv):
        model, obs = self._reddened_pair(true_ebv)
        _, recovered = fit_extinction(model, obs)
        assert abs(recovered - true_ebv) < 0.05, (
            f"E(B-V) recovery failed: expected {true_ebv}, got {recovered:.3f}"
        )

    def test_zero_extinction_gives_low_ebv(self):
        model, obs = self._reddened_pair(0.0)
        _, recovered = fit_extinction(model, obs)
        assert recovered < 0.05

    def test_returns_spectrum_with_matching_wavelength_grid(self):
        model, obs = self._reddened_pair(0.3)
        ext_spec, _ = fit_extinction(model, obs)
        assert hasattr(ext_spec, "wave") and hasattr(ext_spec, "flux")
        assert len(ext_spec.wave) == len(model.wave)


# ── downsample_spec ──────────────────────────────────────────────────────────────

class TestDownsampleSpec:
    def test_output_shape_matches_target(self):
        wave = np.arange(4000.0, 8000.0, 1.0)
        spec = S.ArraySpectrum(wave, np.ones_like(wave), fluxunits="flam")
        target = np.arange(4000.0, 8000.0, 10.0)
        assert downsample_spec(spec, target).shape == target.shape

    def test_flat_spectrum_stays_flat(self):
        wave = np.arange(4000.0, 8000.0, 1.0)
        spec = S.ArraySpectrum(wave, np.ones_like(wave), fluxunits="flam")
        target = np.arange(4100.0, 7900.0, 50.0)
        result = downsample_spec(spec, target)
        assert np.allclose(result, 1.0, atol=0.01)

    def test_coarser_grid_all_finite(self):
        wave = np.arange(4000.0, 7000.0, 0.5)
        spec = S.ArraySpectrum(wave, _blackbody(wave), fluxunits="flam")
        target = np.arange(4100.0, 6900.0, 5.0)
        result = downsample_spec(spec, target)
        assert result.shape == target.shape
        assert np.all(np.isfinite(result))
