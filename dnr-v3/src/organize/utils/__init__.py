import glob
import os
from typing import Any, Optional

import ffmpeg
import numpy as np
import pandas as pd
import soundfile as sf
from omegaconf import DictConfig
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from ...const import AUDIO, CLEANED_DATA_ROOT_VAR, MANIFEST, RAW_DATA_ROOT_VAR


def soundfile_get_file_info_as_dict(file: str) -> dict:
    info = sf.info(file)

    return {
        "samplerate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "format": info.format,
        "subtype": info.subtype,
        "endian": info.endian,
        "format_info": info.format_info,
        "subtype_info": info.subtype_info,
        "sections": info.sections,
        "extra": info.extra_info,
    }


def preface(cfg: DictConfig):
    name = cfg.dataset.name
    subset = cfg.dataset.subset

    raw_data_root = cfg.data.raw_data_root
    clean_data_root = cfg.data.cleaned_data_root

    sampling_rate = cfg.audio.sampling_rate
    bit_depth = cfg.audio.bit_depth
    num_channels = cfg.audio.channels

    print()
    print("".join(["="] * 80))
    print("".join(["="] * 80))
    print(f"Organizing {name} dataset")
    print(f"Subset: {subset}")
    print()
    print(f"Raw data root: {raw_data_root}")
    print(f"Cleaned data root: {clean_data_root}")
    print()
    print(f"Sampling rate: {sampling_rate}")
    print(f"Bit depth: {bit_depth}")
    print(f"Number of channels: {num_channels}")
    print("".join(["="] * 80))
    print("".join(["="] * 80))
    print()


def trim_silence(file: str):
    ext = os.path.splitext(file)[-1].replace(".", "")

    audio = AudioSegment.from_file(file, ext)

    start_trim = detect_leading_silence(audio)
    end_trim = detect_leading_silence(audio.reverse())

    return start_trim, end_trim


def soundfile_get_acodec(bit_depth: int):
    if bit_depth == 16:
        return "PCM_16"
    elif bit_depth == 24:
        return "PCM_24"
    elif bit_depth == 32:
        return "PCM_32"
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")


def soundfile_export(
    segment_audio: np.ndarray,
    segment_path: str,
    cfg: DictConfig,
):
    sampling_rate = cfg.audio.sampling_rate

    acodec = get_acodec_soundfile(cfg.audio.bit_depth)

    os.makedirs(os.path.dirname(segment_path), exist_ok=True)

    sf.write(
        segment_path,
        segment_audio,
        format="wav",
        samplerate=sampling_rate,
        subtype=acodec,
    )


def get_data_list(data_source, cfg):
    glob_path = os.path.join(cfg.data.raw_data_root, data_source.glob)

    print(f"Searching for files in {glob_path}")

    data_list = glob.glob(glob_path, recursive=False)

    print(f"Found {len(data_list)} files for {data_source}")

    df = pd.DataFrame(data_list, columns=["file"])

    return df
