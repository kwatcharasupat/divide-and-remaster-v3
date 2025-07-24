import numpy as np
import pyloudnorm as pyln
from omegaconf import DictConfig


def normalize_audio(
    audio: np.ndarray, target_lufs: float, cfg: DictConfig
) -> np.ndarray:
    """Normalizes an audio array to a target integrated loudness (LUFS).

    Args:
        audio: The input audio data as a NumPy array.
        target_lufs: The desired target LUFS value.
        cfg: The Hydra configuration object.

    Returns:
        A tuple containing:
        - The normalized audio data.
        - The target LUFS value.
        - The original LUFS value of the input audio.
    """
    lufs = get_lufs(audio, cfg=cfg)["loudness_integrated"]

    gain = target_lufs - lufs

    if not np.isfinite(lufs):
        # print("LUFS is not finite", lufs)
        if np.isneginf(lufs):
            return audio, lufs, lufs
        elif np.isposinf(lufs):
            raise ValueError("LUFS is positive infinity")
        else:
            raise ValueError("LUFS is not finite")

    original_lufs = lufs

    audio_out = adjust_audio_lufs(audio, gain)

    return audio_out, target_lufs, original_lufs


def adjust_audio_lufs(audio: np.ndarray, gain: float):
    """Applies a specified gain in dB to an audio array."""
    linear_gain = np.power(10.0, gain / 20.0)
    return audio * linear_gain


def get_lufs(audio: np.ndarray, cfg: DictConfig) -> float:
    """Measures the integrated loudness (LUFS) of an audio array.

    Uses the pyloudnorm library to calculate the LUFS value according to the
    ITU-R BS.1770-4 standard.

    Args:
        audio: The input audio data as a NumPy array.
        cfg: The Hydra configuration object.

    Returns:
        A dictionary containing the integrated loudness value.
    """
    DEFAULT_BLOCK_SIZE = 0.4
    audio_length = audio.shape[-1]
    audio_duration = audio_length / cfg.audio.sampling_rate

    if audio_duration >= 2 * DEFAULT_BLOCK_SIZE:
        block_size = DEFAULT_BLOCK_SIZE
    else:
        # print("Audio duration less than default block size")
        # print("Audio duration: ", audio_duration)
        block_size = (audio_length - 1) / cfg.audio.sampling_rate

    meter = pyln.Meter(cfg.audio.sampling_rate, block_size=block_size)

    loudness_integrated = meter.integrated_loudness(audio.T)

    return {"loudness_integrated": loudness_integrated}
