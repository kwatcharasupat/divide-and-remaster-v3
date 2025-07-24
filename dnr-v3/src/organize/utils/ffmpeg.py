import os
from typing import Dict, List, Optional

import ffmpeg
import numpy as np
from omegaconf import DictConfig

from ...const import RAW_DATA_ROOT_VAR

FFMPEG_CODEC_DICT = {
    16: "pcm_s16le",
    24: "pcm_s24le",
    32: "pcm_s32le",
}


def ffmpeg_get_acodec(bit_depth: int) -> str:
    out = FFMPEG_CODEC_DICT.get(bit_depth, None)

    if out is not None:
        return out

    msg = f"Unsupported bit depth: {bit_depth}"
    raise ValueError(msg)


def ffmpeg_probe_and_clean(
    file: str, output_path: str, loglevel: str = "error"
) -> Optional[str]:
    try:
        ffmpeg.probe(file)
    except ffmpeg.Error as e:
        print(f"Failed to probe {file}")
        print(e)

        if os.path.exists(output_path):
            os.remove(output_path)

        return None

    return file


def ffmpeg_check_output_and_clean(
    output_path: str, cfg: DictConfig, loglevel: str = "error"
) -> Optional[str]:
    try:
        ffmpeg.input(output_path).output(
            "pipe:",
            format="f64le",
            ac=cfg.audio.channels,
            ar=cfg.audio.sampling_rate,
            loglevel="error",
        ).run(capture_stdout=True)
    except ffmpeg.Error as e:
        print(f"Failed to read {output_path}")
        print(e)

        if os.path.exists(output_path):
            os.remove(output_path)

        return None

    return output_path


def _ffmpeg_handle_error(
    e: ffmpeg.Error, src_path: str, output_path: str, cfg: DictConfig
) -> Dict[str, str | None]:
    print(f"Error during CONVERSION: for file: {src_path}")
    print(e.stderr)

    if os.path.exists(output_path):
        os.remove(output_path)

    return {
        "file": src_path.replace(cfg.data.raw_data_root, RAW_DATA_ROOT_VAR),
        "cleaned_path": None,
        "error": str(e),
    }


def ffmpeg_convert(
    src_path: str, output_path: str, cfg: DictConfig, loglevel: str = "error"
) -> None:
    return ffmpeg_convert_auto(
        src_path=src_path, output_path=output_path, cfg=cfg, loglevel=loglevel
    )


def ffmpeg_convert_auto(
    src_path: str, output_path: str, cfg: DictConfig, loglevel: str = "error"
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    acodec = ffmpeg_get_acodec(cfg.audio.bit_depth)

    try:
        (
            ffmpeg.input(src_path)
            .output(
                output_path,
                format="wav",
                acodec=acodec,
                ar=cfg.audio.sampling_rate,
                ac=cfg.audio.channels,
                loglevel=loglevel,
            )
            .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        return _ffmpeg_handle_error(e, src_path, output_path, cfg)


def ffmpeg_convert_split(
    src_path: str,
    output_paths: List[str],
    cfg: DictConfig,
    loglevel: str = "error",
    return_on_first_error: bool = True,
    return_dst_names: bool = False,
) -> None:
    if not return_on_first_error:
        msg = "return_on_first_error must be True"
        raise NotImplementedError(msg)

    acodec = ffmpeg_get_acodec(cfg.audio.bit_depth)

    for idx, output_path in enumerate(output_paths):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            (
                ffmpeg.input(src_path)
                .output(
                    output_path,
                    format="wav",
                    ar=cfg.audio.sampling_rate,
                    ac=1,
                    acodec=acodec,
                    map_channel=f"0.0.{idx}",
                    loglevel=loglevel,
                )
                .run(overwrite_output=True)
            )

        except ffmpeg.Error as e:
            return _ffmpeg_handle_error(e, src_path, output_path, cfg)

    if return_dst_names:
        return output_paths


def ffmpeg_convert_audio_with_downmix(
    src_path: str,
    dst_path: str,
    subset: str,
    split: str,
    cfg: DictConfig,
    loglevel: str = "error",
    original_channels: Optional[int] = None,
    return_dst_names: bool = False,
):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    downmix_mode = cfg.audio.get("downmix_mode", "auto")

    if original_channels is not None and original_channels < 2:
        downmix_mode = "auto"

    if downmix_mode == "auto":
        assert not return_dst_names, "return_dst_names must be False for auto mode"
        return ffmpeg_convert(
            src_path=src_path, output_path=dst_path, cfg=cfg, loglevel=loglevel
        )

    elif downmix_mode in ["split_lr", "lr"]:
        dst_path_right = dst_path.replace(".wav", "_right.wav")
        dst_path_left = dst_path.replace(".wav", "_left.wav")

        return ffmpeg_convert_split(
            src_path=src_path,
            output_paths=[dst_path_left, dst_path_right],
            cfg=cfg,
            loglevel=loglevel,
            return_on_first_error=True,
            return_dst_names=return_dst_names,
        )
    elif downmix_mode == "split_all":
        assert (
            original_channels is not None
        ), "original_channels must be provided for split_all mode"

        output_paths = [
            dst_path.replace(".wav", f"_ch{i}.wav") for i in range(original_channels)
        ]

        return ffmpeg_convert_split(
            src_path=src_path,
            output_paths=output_paths,
            cfg=cfg,
            loglevel=loglevel,
            return_on_first_error=True,
            return_dst_names=return_dst_names,
        )
    else:
        msg = f"Unsupported downmix mode: {downmix_mode}"
        raise ValueError(msg)

    msg = "Unreachable code"
    raise ValueError(msg)


def ffmpeg_reformat_to_buffer(
    file: str,
    sampling_rate: Optional[int] = None,
    num_channels: Optional[int] = None,
    intermediate_format: str = "f64le",
    channel_first: bool = False,
) -> np.ndarray:
    if num_channels is None:
        num_channels = 1

    kwargs = {"format": intermediate_format, "ac": num_channels, "loglevel": "error"}

    if sampling_rate is not None:
        kwargs["ar"] = sampling_rate

    out, err = ffmpeg.input(file).output("pipe:", **kwargs).run(capture_stdout=True)

    audio = np.frombuffer(out, dtype=np.float64).reshape(-1, num_channels)

    if channel_first:
        audio = audio.T

    return audio


def ffmpeg_mix_audio(
    src_paths: List[str],
    dst_path: str,
    cfg: DictConfig,
    loglevel: str = "error",
    ac: int = None,
) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    acodec = ffmpeg_get_acodec(cfg.audio.bit_depth)

    input_streams = [ffmpeg.input(src_path) for src_path in src_paths]

    try:
        ffmpeg.filter(
            input_streams,
            "amix",
            inputs=len(input_streams),
            dropout_transition=0,
            duration="longest",
        ).output(
            dst_path,
            ac=cfg.audio.channels if ac is None else ac,
            ar=cfg.audio.sampling_rate,
            acodec=acodec,
            loglevel=loglevel,
        ).run(overwrite_output=True)
    except ffmpeg.Error as e:
        return _ffmpeg_handle_error(e, src_paths, dst_path, cfg)
