# ============================================================
# FILE: tests/unit/test_preprocessing.py
# ============================================================
import numpy as np

from src.ml.preprocessing.signal import (
    bandpass_filter,
    normalize_signal,
    preprocess_ecg,
)


class TestSignalPreprocessing:
    """Tests for ECG signal preprocessing functions."""

    def test_bandpass_filter_shape_preserved(self):
        """Filter should preserve signal shape."""
        signal = np.random.randn(12, 5000)
        filtered = bandpass_filter(signal)
        assert filtered.shape == signal.shape

    def test_normalize_signal_zero_mean(self):
        """Normalized signal should have zero mean per lead."""
        signal = np.random.randn(12, 5000) * 10 + 5
        normalized = normalize_signal(signal)

        means = normalized.mean(axis=1)
        np.testing.assert_array_almost_equal(means, np.zeros(12), decimal=10)

    def test_normalize_signal_unit_variance(self):
        """Normalized signal should have unit variance per lead."""
        signal = np.random.randn(12, 5000) * 10 + 5
        normalized = normalize_signal(signal)

        stds = normalized.std(axis=1)
        np.testing.assert_array_almost_equal(stds, np.ones(12), decimal=10)

    def test_preprocess_ecg_resampling(self):
        """Preprocessing should resample to target frequency."""
        signal = np.random.randn(12, 5000)  # 10 seconds at 500Hz
        processed = preprocess_ecg(signal, fs=500, target_fs=250)

        # Should be 2500 samples at 250Hz
        assert processed.shape == (12, 2500)

    def test_preprocess_ecg_handles_flat_signal(self):
        """Preprocessing should handle constant signal without error."""
        signal = np.ones((12, 5000))
        processed = preprocess_ecg(signal)

        # Should not contain NaN or Inf
        assert np.isfinite(processed).all()
