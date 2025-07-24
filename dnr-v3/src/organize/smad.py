from enum import StrEnum

import pandas as pd


class SpeechMusicActivations(StrEnum):
    NOTHING = "nothing"
    SPEECH = "vox-only"
    MUSIC = "music-only"
    SPEECH_MUSIC = "vox-music"


def index_to_activation_class(index):
    """Converts a numerical class index to a SpeechMusicActivations enum."""
    if index == 0:
        return SpeechMusicActivations.NOTHING
    elif index == 1:
        return SpeechMusicActivations.SPEECH
    elif index == 2:
        return SpeechMusicActivations.MUSIC
    elif index == 3:
        return SpeechMusicActivations.SPEECH_MUSIC
    else:
        raise ValueError(f"Unsupported index: {index}")


def class_to_shorthand(class_):
    """Converts a SpeechMusicActivations enum to its two-letter shorthand."""
    if class_ == SpeechMusicActivations.NOTHING:
        return "no"
    elif class_ == SpeechMusicActivations.SPEECH:
        return "so"
    elif class_ == SpeechMusicActivations.MUSIC:
        return "mo"
    elif class_ == SpeechMusicActivations.SPEECH_MUSIC:
        return "sm"
    else:
        raise ValueError(f"Unsupported class: {class_}")


def get_segments_from_activation(activations, frame_size_seconds):
    """
    Converts a frame-by-frame activation array into a list of timed segments.

    This is used to transform the output of a speech/music activity detector
    into a structured format with start times, end times, and class labels.
    """
    segments = []

    n_frames = activations.shape[-1]

    current_segment = activations[0]
    start_index = 0

    for i in range(1, n_frames):
        if activations[i] != current_segment:
            segments.append(
                {
                    "start_time": start_index * frame_size_seconds,
                    "end_time": i * frame_size_seconds,
                    "class": index_to_activation_class(current_segment),
                }
            )
            current_segment = activations[i]
            start_index = i

    segments.append(
        {
            "start_time": start_index * frame_size_seconds,
            "end_time": n_frames * frame_size_seconds,
            "class": index_to_activation_class(current_segment),
        }
    )

    segments = pd.DataFrame(segments)

    return segments
