# ============================================================
# FILE: src/ml/preprocessing/signal.py
# ============================================================
import numpy as np
from scipy import signal as scipy_signal


def bandpass_filter(
    ecg_signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: int = 500,
    order: int = 4,
) -> np.ndarray:
    """
    Apply bandpass filter to remove noise from ECG signal.

    Args:
        ecg_signal: ECG signal array, shape (n_leads, n_samples)
        lowcut: Low frequency cutoff (Hz)
        highcut: High frequency cutoff (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order

    Returns:
        Filtered ECG signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = scipy_signal.butter(order, [low, high], btype="band")

    filtered = np.zeros_like(ecg_signal)
    for i in range(ecg_signal.shape[0]):
        filtered[i] = scipy_signal.filtfilt(b, a, ecg_signal[i])

    return filtered


def normalize_signal(ecg_signal: np.ndarray) -> np.ndarray:
    """
    Normalize ECG signal to zero mean and unit variance per lead.

    Args:
        ecg_signal: ECG signal array, shape (n_leads, n_samples)

    Returns:
        Normalized ECG signal
    """
    mean = ecg_signal.mean(axis=1, keepdims=True)
    std = ecg_signal.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)  # Avoid division by zero

    return (ecg_signal - mean) / std


def preprocess_ecg(
    ecg_signal: np.ndarray,
    fs: int = 500,
    target_fs: int = 250,
) -> np.ndarray:
    """
    Full preprocessing pipeline for ECG signal.

    Args:
        ecg_signal: Raw ECG signal, shape (n_leads, n_samples)
        fs: Original sampling frequency
        target_fs: Target sampling frequency

    Returns:
        Preprocessed ECG signal
    """
    # 1. Bandpass filter
    filtered = bandpass_filter(ecg_signal, fs=fs)

    # 2. Resample if needed
    if fs != target_fs:
        num_samples = int(filtered.shape[1] * target_fs / fs)
        resampled = np.zeros((filtered.shape[0], num_samples))
        for i in range(filtered.shape[0]):
            resampled[i] = scipy_signal.resample(filtered[i], num_samples)
        filtered = resampled

    # 3. Normalize
    normalized = normalize_signal(filtered)

    return normalized
