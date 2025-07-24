import json
import tempfile

import ffmpeg
import numpy as np
import soundfile as sf
from pedalboard import Limiter

# Loosely based on https://gist.github.com/bastibe/747283c55aad66404046


def peak_limit(audio, threshold_db, sample_rate, release_ms: float = 100.0):
    """Applies a simple peak limiter to the audio."""
    limiter = Limiter(threshold_db=threshold_db, release_ms=release_ms)
    peak_limited_audio = limiter(audio, sample_rate)

    return peak_limited_audio


TARGET_LUFS_MIN = -70.0
TARGET_LUFS_MAX = -5.0


def true_peak_limit(
    audio,
    target_lufs,
    threshold_db,
    sample_rate,
    release_ms: float = 100.0,
    upsampling_factor: int = 4,
):
    """
    Applies a true-peak limiter to the audio using ffmpeg's loudnorm filter.

    This is a two-pass approach that first analyzes the audio to determine the
    necessary parameters and then applies the loudnorm filter to normalize
    loudness and limit the true peak.
    """
    if not (TARGET_LUFS_MIN <= target_lufs <= TARGET_LUFS_MAX):
        print(
            f"Target LUFS {target_lufs} is not in the range [{TARGET_LUFS_MIN}, {TARGET_LUFS_MAX}]"
        )

        target_lufs = np.clip(target_lufs, TARGET_LUFS_MIN, TARGET_LUFS_MAX)

    n_channels = audio.shape[0]

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        sf.write(temp_file.name, audio.T, sample_rate)

        try:
            out, err = (
                ffmpeg.input(temp_file.name)
                .filter("loudnorm", i=target_lufs, tp=threshold_db, print_format="json")
                .output("pipe:", format="null")
                .run(capture_stdout=True, capture_stderr=True)
            )

            err = err.decode("utf-8")

            info = "{" + err.split("{")[1]

            info = json.loads(info)
        except ffmpeg.Error as e:
            out, err = (
                ffmpeg.input(temp_file.name)
                .filter("loudnorm", i=target_lufs, tp=threshold_db, print_format="json")
                .output("pipe:", format="null", loglevel="info")
                .run()
            )

            raise e

        try:
            out, err = (
                ffmpeg.input(temp_file.name)
                .filter(
                    "loudnorm",
                    linear=False if info["normalization_type"] == "dynamic" else True,
                    i=info["input_i"],
                    tp=threshold_db,
                    measured_i=info["input_i"],
                    measured_lra=info["input_lra"],
                    measured_tp=info["input_tp"],
                    measured_thresh=info["input_thresh"],
                    offset=info["target_offset"],
                    print_format="json",
                )
                .output(
                    "pipe:",
                    format="f64le",
                    ar=sample_rate,
                    ac=n_channels,
                    loglevel="error",
                )
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error:
            out, err = (
                ffmpeg.input(temp_file.name)
                .filter(
                    "loudnorm",
                    linear=False if info["normalization_type"] == "dynamic" else True,
                    i=info["input_i"],
                    tp=threshold_db,
                    measured_i=info["input_i"],
                    measured_lra=info["input_lra"],
                    measured_tp=info["input_tp"],
                    measured_thresh=info["input_thresh"],
                    offset=info["target_offset"],
                    print_format="json",
                )
                .output(
                    "pipe:",
                    format="f64le",
                    ar=sample_rate,
                    ac=n_channels,
                    loglevel="info",
                )
                .run()
            )

        out = np.frombuffer(out, dtype=np.float64).reshape(-1, n_channels).T

    return out


def simple_peak_limiter(
    audio, peak_threshold_db, sample_rate, release_ms: float = 100.0
):
    """A simple implementation of a peak limiter using pedalboard."""
    limiter = Limiter(threshold_db=peak_threshold_db, release_ms=release_ms)

    peak_limited_audio = limiter(audio, sample_rate)

    abs_audio = np.abs(audio)
    abs_peak_limited_audio = np.abs(peak_limited_audio)

    limited = abs_audio > abs_peak_limited_audio

    assert np.all(abs_audio[limited] > 0)

    safe_audio = np.where(abs_audio == 0, 1.0, audio)

    gain = np.where(
        abs_audio > abs_peak_limited_audio, peak_limited_audio / safe_audio, 1.0
    )

    assert np.all(np.isfinite(gain))
    assert np.all(gain >= 0)
    assert np.all(gain <= 1)

    return gain


if __name__ == "__main__":
    import soundfile as sf

    file = "/root/data/cass-data/dnr-rebuild/eval/142/mix.wav"

    audio, sr = sf.read(file)

    gain = simple_peak_limiter(audio * 2, -2, sr)

    import matplotlib.pyplot as plt

    time = np.arange(len(audio)) / sr

    f, ax = plt.subplots(2, 1, figsize=(60, 6))
    ax[0].plot(time, audio, label="Original")
    ax[0].plot(time, audio * gain, label="Limited")

    gain_db = 20 * np.log10(gain)

    ax[1].plot(time, gain_db, label="Gain (dB)")

    ax[0].legend()

    plt.savefig("limiter.pdf")

    sf.write("limited.wav", audio * gain, sr)
