from typing import Dict

import numpy as np
from omegaconf import DictConfig

from .loudness import get_lufs
from .peak import get_naive_peak, get_true_peak


def get_audio_params(audio: np.ndarray, cfg: DictConfig) -> Dict[str, float]:
    """Calculates and returns a dictionary of key audio parameters."""
    lufs_dict = get_lufs(audio, cfg=cfg)

    true_peak_dict = get_true_peak(audio, cfg=cfg)
    true_peak_dict.pop("oversampled_audio")

    naive_peak_dict = get_naive_peak(audio)

    return {**lufs_dict, **true_peak_dict, **naive_peak_dict}
