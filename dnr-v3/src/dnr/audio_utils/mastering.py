import os
from typing import Dict

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ...const import MIXTURE
from ..random import _get_mix_lufs
from .limiter import true_peak_limit
from .loudness import normalize_audio
from .misc import get_audio_params
from .peak import get_naive_peak, get_true_peak


def apply_mastering(
    audio_dict: Dict[str, np.ndarray],
    lufs_dict: Dict[str, float],
    cfg: DictConfig,
    random_state: np.random.Generator,
    compensation_gain: float = 0.0,
    idx: int = None,
    split: str = None,
) -> Dict[str, np.ndarray]:
    """Applies the final mastering process to a collection of submixes.

    This function normalizes the full mixture to a target LUFS, applies the same
    gain to all submixes, and then applies peak correction/limiting as specified
    in the configuration.

    Args:
        audio_dict: A dictionary of all submix audio arrays.
        lufs_dict: A dictionary of the target LUFS for each submix.
        cfg: The Hydra configuration object.
        random_state: The random number generator.
        compensation_gain: An optional gain to apply.
        idx: The index of the current file (for precomputed loudness).
        split: The name of the current split (for precomputed loudness).

    Returns:
        A tuple containing:
        - A dictionary of the mastered audio for all submixes and the final mixture.
        - A dictionary of the final audio parameters for each mastered track.
    """
    if cfg.loudness.master.distr == "precomputed":
        assert idx is not None
        assert split is not None
        lufs_file = cfg.loudness.master.file.format(split=split)
        lufs_df = pd.read_csv(os.path.expandvars(lufs_file)).set_index("idx", drop=True)
        target_mix_lufs = lufs_df.iloc[idx][cfg.loudness.master.column].item()
    else:
        target_mix_lufs = _get_mix_lufs(cfg=cfg, random_state=random_state)

    audio_dict_mastered = {}

    mixture = audio_dict[MIXTURE]

    mixture_normalized, mixture_lufs, original_lufs = normalize_audio(
        mixture,
        target_lufs=target_mix_lufs,
        cfg=cfg,
    )

    master_gain = mixture_lufs - original_lufs
    linear_gain = np.power(10.0, master_gain / 20.0)

    for submix in audio_dict:
        if submix == MIXTURE:
            continue
        audio_dict_mastered[submix] = audio_dict[submix] * linear_gain

        if cfg.loudness.peak.correction_mode == "limiter":
            audio_dict_mastered[submix] = true_peak_limit(
                audio_dict_mastered[submix],
                lufs_dict[submix] + master_gain,
                cfg.loudness.peak.threshold,
                cfg.audio.sampling_rate,
            )

            peak = get_naive_peak(audio_dict_mastered[submix])["naive_peak"]

            if peak > 0:
                raise ValueError(f"Peak clipped for {submix} at {peak} dB")

    mixture_normalized = sum(audio_dict_mastered.values())

    if cfg.loudness.peak.correction_mode == "global":
        if cfg.loudness.peak.detector == "true-peak":
            true_peak_outputs = get_true_peak(mixture_normalized, cfg)
            current_true_peak = true_peak_outputs["true_peak"]
            oversampled_audio = true_peak_outputs["oversampled_audio"]
        else:
            current_true_peak = get_naive_peak(mixture_normalized)["naive_peak"]
            oversampled_audio = None

        max_true_peak = cfg.loudness.peak.threshold
        ducking_gain = current_true_peak - max_true_peak

        if ducking_gain < 0:
            linear_ducking_gain = np.power(10.0, ducking_gain / 20.0)
            for submix in audio_dict:
                if submix == MIXTURE:
                    continue
                audio_dict_mastered[submix] = audio_dict[submix] * linear_ducking_gain
    elif cfg.loudness.peak.correction_mode == "limiter":
        pass
    else:
        raise NotImplementedError("Other peak modes not implemented")

    audio_dict_mastered[MIXTURE] = sum(audio_dict_mastered.values())

    audio_params = {}
    for key in audio_dict_mastered:
        audio_params[key] = get_audio_params(audio_dict_mastered[key], cfg=cfg)

        if audio_params[key]["true_peak"] > 0:
            print(f"True peak clipped for {key} at {audio_params[key]['true_peak']} dB")

            if audio_params[key]["naive_peak"] > 0:
                print(
                    f"Naive peak also clipped for {key} at {audio_params[key]['naive_peak']} dB"
                )

    # assert np.isclose(audio_params[MIXTURE]["loudness_integrated"], target_mix_lufs, atol=0.1)
    # print(f"Target LUFS: {target_mix_lufs} dB", f"Actual LUFS: {audio_params[MIXTURE]['loudness_integrated']} dB")
    # print(f"LUFS Delta: {target_mix_lufs - audio_params[MIXTURE]['loudness_integrated']:.2f} dB")

    return audio_dict_mastered, audio_params
