import os
from typing import Dict

import numpy as np
import pandas as pd
import soundfile as sf
from omegaconf import DictConfig

from ..const import AUDIO, AUDIO_METADATA, CLEANED_DATA_ROOT_VAR, MANIFEST, MIXTURE


def output(
    final_audio_dict: Dict[str, np.ndarray],
    final_annots_dict: Dict[str, pd.DataFrame],
    final_audio_param_dict: Dict[str, pd.DataFrame],
    idx: int,
    split: str,
    submix: str,
    subset: str,
    cfg: DictConfig,
):
    """Saves all generated audio files, manifests, and metadata to disk.

    Args:
        final_audio_dict: Dictionary containing the audio data for each submix and the
                          final mixture.
        final_annots_dict: Dictionary containing the annotation DataFrames for each
                           submix.
        final_audio_param_dict: Dictionary containing audio parameters (LUFS, peak)
                                for each generated audio file.
        idx: The index of the current generated file.
        split: The name of the current dataset split (e.g., "train").
        submix: The name of the submix (used for directory structure).
        subset: The name of the data subset (used for directory structure).
        cfg: The Hydra configuration object.
    """
    audio_dir = os.path.join(cfg.output_dir, AUDIO, subset, split, f"{idx:06d}")
    os.makedirs(audio_dir, exist_ok=True)

    annots_dir = os.path.join(cfg.output_dir, MANIFEST, subset, split, f"{idx:06d}")
    os.makedirs(annots_dir, exist_ok=True)

    audio_param_dir = os.path.join(
        cfg.output_dir,
        AUDIO_METADATA,
        subset,
        split,
    )
    os.makedirs(audio_param_dir, exist_ok=True)

    if cfg.audio.bit_depth == 16:
        subtype = "PCM_16"
    elif cfg.audio.bit_depth == 24:
        subtype = "PCM_24"
    else:
        raise ValueError(f"Unsupported bit depth: {cfg.audio.bit_depth}")

    for submix in final_audio_dict:
        if submix == "mixture" and not cfg.get("output_mixture", True):
            continue

        audio_path = os.path.join(audio_dir, f"{submix}.wav")
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        sf.write(
            audio_path,
            final_audio_dict[submix].T,
            cfg.audio.sampling_rate,
            subtype=subtype,
        )

        if submix in [MIXTURE] + list(cfg.composite.keys()):
            continue

        # print(final_annots_dict)

        df_annots = pd.DataFrame(final_annots_dict[submix])
        df_annots["file"] = df_annots["file"].apply(
            lambda x: x.replace(cfg.data_root, CLEANED_DATA_ROOT_VAR)
        )

        df_annots.to_csv(os.path.join(annots_dir, f"{submix}.csv"), index=False)

        # print(final_audio_param_dict)

    df_params = pd.DataFrame(final_audio_param_dict).T

    df_params.to_csv(os.path.join(audio_param_dir, f"{idx:06d}.csv"), index=True)
