import numpy as np
import resampy
from omegaconf import DictConfig


def get_true_peak(audio: np.ndarray, cfg: DictConfig) -> float:
    """Calculates the true-peak level of an audio array by oversampling."""
    return get_true_peak_numpy(audio, cfg)


def get_naive_peak(audio: np.ndarray) -> float:
    """Calculates the sample-based ('naive') peak level of an audio array."""
    peak_db = _get_peak(audio)

    return {"naive_peak": peak_db}


def _get_peak(audio: np.ndarray, eps=1e-8):
    """Helper function to find the peak dB level of an audio array."""
    peak_linear = np.max(np.abs(audio))
    peak_db = 20 * np.log10(peak_linear + eps)
    return peak_db


def get_true_peak_numpy(audio: np.ndarray, cfg: DictConfig) -> float:
    """Numpy-based implementation of a true-peak detector."""
    # reimplementation of the essentia TruePeakDetector

    oversampled_audio = resampy.resample(
        audio,
        cfg.audio.sampling_rate,
        int(cfg.audio.sampling_rate * cfg.loudness.peak.oversampling_factor),
        filter="kaiser_fast",
    )

    true_peak_db = _get_peak(oversampled_audio)

    return {
        "true_peak": true_peak_db,
        "oversampled_audio": oversampled_audio,
    }
