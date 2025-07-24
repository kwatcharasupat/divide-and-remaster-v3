import os
import shutil

import ffmpeg
import numpy as np
import soundfile as sf
from tqdm.contrib.concurrent import process_map

AUDIO = "audio"
NPY32 = "npy32"

SPLIT_MAP = {
    "train": "tr",
    "val": "cv",
    "test": "tt",
}

STEM_MAP = {
    "speech": "speech",
    "music": "music",
    "sfx": "sfx",
    "mixture": "mix",
}


def refactor_file(file, src_split_dir, dst_dir, subset_name, split_dst, resample=None):
    os.makedirs(
        os.path.join(dst_dir, AUDIO, subset_name, split_dst, file), exist_ok=True
    )

    for stem_dst, stem_src in STEM_MAP.items():
        src_path = os.path.join(src_split_dir, file, f"{stem_src}.wav")
        dst_path = os.path.join(
            dst_dir, AUDIO, subset_name, split_dst, file, f"{stem_dst}.wav"
        )

        if resample is not None:
            (
                ffmpeg.input(src_path)
                .output(dst_path, ar=resample, loglevel="error")
                .run(overwrite_output=True)
            )

        else:
            shutil.copy(src_path, dst_path)

        audio, fs = sf.read(dst_path, always_2d=True, dtype="float64")

        assert fs == resample

        audio = audio.T

        assert audio.shape[-1] == fs * 60

        npy64_file = dst_path.replace(".wav", ".npy").replace(AUDIO, NPY32)

        os.makedirs(os.path.dirname(npy64_file), exist_ok=True)

        np.save(npy64_file, audio)


def refactor_v2(
    src_dir="/root/data/cass-data/dnr-v2-original",
    dst_dir="/fsx_vfx/projects/aa-cass/data/dnr-variants/dnr-v2",
    subset_name="44k",
    resample=44100,
):
    if subset_name == "48k":
        assert resample == 48000

    if subset_name == "44k":
        assert resample is None or resample == 44100

    os.makedirs(dst_dir, exist_ok=True)

    os.makedirs(os.path.join(dst_dir, AUDIO), exist_ok=True)

    os.makedirs(os.path.join(dst_dir, AUDIO, subset_name), exist_ok=True)

    for split_dst, split_src in SPLIT_MAP.items():
        os.makedirs(os.path.join(dst_dir, AUDIO, subset_name, split_dst), exist_ok=True)

        src_split_dir = os.path.join(src_dir, "dnr_v2", split_src)

        files = os.listdir(src_split_dir)
        files = [f for f in files if os.path.isdir(os.path.join(src_split_dir, f))]

        process_map(
            refactor_file,
            files,
            [src_split_dir] * len(files),
            [dst_dir] * len(files),
            [subset_name] * len(files),
            [split_dst] * len(files),
            [resample] * len(files),
            chunksize=1,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(refactor_v2)
