import os
from typing import Optional

from omegaconf import DictConfig

from ...const import AUDIO, CLEANED_DATA_ROOT_VAR, RAW_DATA_ROOT_VAR
from .ffmpeg import (
    ffmpeg_check_output_and_clean,
    ffmpeg_convert,
    ffmpeg_probe_and_clean,
)


def _generic_organize_file(
    *,
    src_path: str,
    split: str,
    subset: str,
    cfg: DictConfig,
    langcode: Optional[str] = None,
    langname: Optional[str] = None,
    speaker_gender: Optional[str] = None,
    speaker_id: Optional[str] = None,
    probe_input: bool = False,
    probe_output: bool = False,
    filename_override: Optional[str] = None,
    **kwargs,  # noqa: ANN003
):
    file = os.path.expandvars(src_path)

    if filename_override is not None:
        filename = filename_override
    else:
        filename = os.path.basename(src_path)

    output_path = os.path.join(
        cfg.data.cleaned_data_root,
        cfg.dataset.name,
        AUDIO,
        subset,
        split,
        filename,
    )

    if probe_input:
        file = ffmpeg_probe_and_clean(file, output_path)
        if file is None:
            return None

    ffmpeg_convert(
        src_path=file,
        output_path=output_path,
        cfg=cfg,
    )

    if probe_output:
        ret = ffmpeg_check_output_and_clean(output_path, cfg)
        if ret is None:
            return None

    manifest_entry = {
        "file": src_path.replace(cfg.data.raw_data_root, RAW_DATA_ROOT_VAR),
        "cleaned_path": output_path.replace(
            cfg.data.cleaned_data_root, CLEANED_DATA_ROOT_VAR
        ),
        "subset": subset,
        "split": split,
    }

    for key, item in [
        ("langcode", langcode),
        ("language", langname),
        ("speaker_gender", speaker_gender),
        ("dsid_speaker_id", speaker_id),
    ]:
        if item is not None:
            manifest_entry[key] = item

    return manifest_entry
