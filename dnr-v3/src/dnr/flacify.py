import glob
import os

import ffmpeg
from tqdm.contrib.concurrent import process_map


def compress_file(
    src_path,
):
    dst_path = src_path.replace(".wav", ".flac").replace("/audio/", "/flac/")

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    (
        ffmpeg.input(src_path)
        .output(dst_path, compression_level=12, loglevel="error")
        .run(overwrite_output=True)
    )


def run_flacify(
    src_dir,
):
    for variant in sorted(os.listdir(src_dir)):
        if not os.path.isdir(f"{src_dir}/{variant}"):
            continue

        if variant in ["spa", "yor", "bnt", "multi", "eus", "inc", "dra", "jpn"]:
            continue

        print(f"Processing {variant}")
        files = glob.glob(f"{src_dir}/{variant}/**/*.wav", recursive=True)

        process_map(compress_file, sorted(files), chunksize=8)


if __name__ == "__main__":
    import fire

    fire.Fire(run_flacify)
