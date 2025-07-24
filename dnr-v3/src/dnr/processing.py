from typing import Tuple

import numpy as np
from omegaconf import DictConfig

from .random import _get_length_seconds, _get_start_seconds


def get_time_range(
    event_length_seconds: float,
    min_start_seconds: float,
    submix_duration_seconds: float,
    submix: str,
    random_state: np.random.Generator,
    cfg: DictConfig,
    verbose: bool = False,
) -> Tuple[int, int]:
    """Calculates the start time and duration for an audio event within a submix.

    This function determines a valid time range for an audio segment, ensuring it
    adheres to the minimum duration and placement constraints defined in the
    configuration.

    Args:
        event_length_seconds: The total duration of the source audio file.
        min_start_seconds: The earliest possible start time for the event.
        submix_duration_seconds: The total duration of the submix timeline.
        submix: The name of the submix being generated (e.g., "speech", "music").
        random_state: The random number generator.
        cfg: The Hydra configuration object.
        verbose: If True, prints additional debug information.

    Returns:
        A tuple containing the start sample and the length in samples, or (None, None)
        if no valid placement can be found.
    """
    remaining_seconds = submix_duration_seconds - min_start_seconds

    min_prop = cfg.events[submix].min_duration_prop
    min_seconds = cfg.events[submix].get("min_duration_seconds", 0.0)

    min_duration_seconds = max(min_prop * event_length_seconds, min_seconds)

    full_segment_only = min_prop == 1.0

    if remaining_seconds < min_duration_seconds:
        return None, None

    max_start_seconds = submix_duration_seconds - min_duration_seconds

    if not full_segment_only:
        if cfg.events[submix].min_time_before_end_seconds is not None:
            max_start_seconds_ = (
                submix_duration_seconds - cfg.events[submix].min_time_before_end_seconds
            )
            max_start_seconds = min(max_start_seconds, max_start_seconds_)

        if cfg.events[submix].min_time_before_end_prop is not None:
            max_start_seconds_ = (
                submix_duration_seconds
                - cfg.events[submix].min_time_before_end_prop * event_length_seconds
            )
            max_start_seconds = min(max_start_seconds, max_start_seconds_)

        if max_start_seconds < min_start_seconds:
            return None, None

    start_seconds = _get_start_seconds(
        min_start_seconds=min_start_seconds,
        max_start_seconds=max_start_seconds,
        submix=submix,
        cfg=cfg,
        random_state=random_state,
    )

    if (
        cfg.events[submix].get("full_segment_only", False)
        or event_length_seconds <= min_duration_seconds
    ):
        length_seconds = event_length_seconds
        start_seconds = max(min(start_seconds, max_start_seconds), min_start_seconds)
    else:
        if submix_duration_seconds - start_seconds > event_length_seconds:
            max_length_seconds = event_length_seconds
        else:
            max_length_seconds = submix_duration_seconds - start_seconds

        if max_length_seconds < 0:
            return None, None

        length_seconds = _get_length_seconds(
            max_length_seconds=max_length_seconds,
            min_length_seconds=min_duration_seconds,
            submix=submix,
            cfg=cfg,
            random_state=random_state,
        )

    start_sample = int(start_seconds * cfg.audio.sampling_rate)
    length_sample = int(length_seconds * cfg.audio.sampling_rate)

    if length_sample <= 0:
        return None, None

    assert start_sample >= 0
    assert length_sample > 0
    assert length_sample <= event_length_seconds * cfg.audio.sampling_rate
    assert (
        start_sample + length_sample
        <= submix_duration_seconds * cfg.audio.sampling_rate
    )

    return start_sample, length_sample


def load_and_select_segment(
    file_path: str,
    length_sample: int,
    random_segment_start: bool,
    random_state: np.random.Generator,
) -> np.ndarray:
    """Loads an audio file and selects a segment of a given length.

    Args:
        file_path: Path to the NPY audio file.
        length_sample: The desired length of the segment in samples.
        random_segment_start: If True, select a random start point for the segment.
                              If False, start from the beginning.
        random_state: The random number generator.

    Returns:
        The selected audio segment as a NumPy array.
    """
    audio = np.load(file_path, mmap_mode="r")
    audio_length_samples = audio.shape[-1]

    if random_segment_start:
        # Ensure the segment doesn't go out of bounds
        max_start = audio_length_samples - length_sample
        if max_start < 0:
            # This can happen if the file is shorter than the desired length
            segment_start_sample = 0
            length_sample = audio_length_samples
        else:
            segment_start_sample = random_state.integers(0, max_start)
    else:
        segment_start_sample = 0

    audio_segment = audio[
        :, segment_start_sample : segment_start_sample + length_sample
    ]
    return audio_segment, segment_start_sample


def process_stems(
    audio_segment_dict: dict,
    submix_lufs: float,
    cfg: DictConfig,
) -> Tuple[dict, dict, float]:
    """Normalizes and adjusts the loudness of a dictionary of audio stems.

    Args:
        audio_segment_dict: A dictionary where keys are stem names and values are
                            the corresponding audio data as NumPy arrays.
        submix_lufs: The target loudness for the combined submix.
        cfg: The Hydra configuration object.

    Returns:
        A tuple containing:
        - The processed audio segment dictionary.
        - A dictionary with the final LUFS value for each stem.
        - The effective gain applied to the entire clip.
    """
    clip_audio_segment = sum(audio_segment_dict.values())
    _, clip_lufs, original_lufs = normalize_audio(
        clip_audio_segment, submix_lufs, cfg=cfg
    )

    effective_clip_gain = clip_lufs - original_lufs

    final_audio_segments = {}
    final_clip_lufs = {}

    for substem, segment in audio_segment_dict.items():
        final_audio_segments[substem] = adjust_audio_lufs(
            segment, effective_clip_gain
        )
        final_clip_lufs[substem] = get_lufs(final_audio_segments[substem], cfg=cfg)[
            "loudness_integrated"
        ]

    return final_audio_segments, final_clip_lufs, effective_clip_gain


def _create_audio_event(
    get_and_check_func, max_trials, **kwargs
) -> Tuple[dict, bool]:
    """Attempts to generate a valid audio event within a set number of trials.

    Args:
        get_and_check_func: The function to call for generating the event
                            (e.g., `get_file_and_check`).
        max_trials: The maximum number of times to attempt generation.
        **kwargs: Arguments to pass to the generation function.

    Returns:
        A tuple containing:
        - A dictionary with the results from the generation function.
        - A boolean indicating if the generation was successful.
    """
    for _ in range(max_trials):
        results = get_and_check_func(**kwargs)
        # The tuple's structure varies, but a valid start_sample is a good indicator
        if results[2] is not None:
            # Check for finite LUFS values
            lufs = results[-1]  # clip_lufs or clip_lufs_dict
            if isinstance(lufs, dict):
                if np.isfinite(list(lufs.values())).all():
                    return results, True
            elif np.isfinite(lufs):
                return results, True

    return None, False