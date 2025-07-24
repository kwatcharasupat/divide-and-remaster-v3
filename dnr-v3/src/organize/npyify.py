import glob
import os

import numpy as np
import soundfile as sf
from tqdm.contrib.concurrent import process_map

from ..const import AUDIO, NPY32, NPY64


def convert_to_npy(audio_file: str, save_format: str = "both", check_length=48000 * 60):
    """Converts a single audio file (.wav) to a NumPy (.npy) file."""
    try:
        audio, sr = sf.read(audio_file, always_2d=True, dtype="float64")
        # audio = ffmpeg_reformat_to_buffer(audio_file)
    except Exception as e:
        print(f"Failed to read {audio_file}")
        print(e)
        return

    audio = audio.T  # (n_samples, n_channels) -> (n_channels, n_samples)

    if check_length is not None:
        assert (
            audio.shape[1] == check_length
        ), f"Audio length mismatch: {audio.shape[1]} != {check_length}"

    if save_format in ["both", "npy64"]:
        npy64_file = audio_file.replace(AUDIO, NPY64).replace(".wav", ".npy")
        os.makedirs(os.path.dirname(npy64_file), exist_ok=True)
        np.save(npy64_file, audio)

    if save_format in ["both", "npy32"]:
        npy32_file = audio_file.replace(AUDIO, NPY32).replace(".wav", ".npy")
        os.makedirs(os.path.dirname(npy32_file), exist_ok=True)
        audio = audio.astype(np.float32)
        np.save(npy32_file, audio)


def check_npy_exists(audio_file: str, save_format: str = "npy64"):
    """Checks if a .npy file already exists for a given audio file."""
    assert (
        save_format in ["npy64", "npy32"]
    ), f"Invalid save_format: {save_format} (must be 'npy64' or 'npy32'; 'both' is not allowed here)"

    if save_format in ["both", "npy64"]:
        npy64_file = audio_file.replace(AUDIO, NPY64).replace(".wav", ".npy")
        if not os.path.exists(npy64_file):
            print(f"Missing {npy64_file}")
            return audio_file

    if save_format in ["both", "npy32"]:
        npy32_file = audio_file.replace(AUDIO, NPY32).replace(".wav", ".npy")
        if not os.path.exists(npy32_file):
            print(f"Missing {npy32_file}")
            return audio_file

    return None


def npyify(
    audio_root: str,
    save_format: str = "npy64",
    check_length=0,
    overwrite_glob=False,
    check_only=False,
):
    """
    Converts all .wav files in a directory to .npy format in parallel.

    This function can also be used to check for missing .npy files without
    performing the conversion.
    """
    if check_length <= 0:
        check_length = None

    if overwrite_glob:
        glob_str = audio_root
    else:
        glob_str = f"{audio_root}/**/*.wav"

    audio_files = glob.glob(glob_str, recursive=True)

    if check_only:
        print(f"Checking {len(audio_files)} files")
        missings = process_map(
            check_npy_exists, audio_files, [save_format] * len(audio_files), chunksize=1
        )

        missings = [m for m in missings if m is not None]

        if len(missings) > 0:
            print("Missing {} files".format(len(missings)))
            print("\n".join(missings))

    else:
        process_map(
            convert_to_npy,
            audio_files,
            [save_format] * len(audio_files),
            [check_length] * len(audio_files),
            chunksize=1,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(npyify)
