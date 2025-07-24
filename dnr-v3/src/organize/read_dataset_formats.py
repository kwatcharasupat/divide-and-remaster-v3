import glob
import os

import pandas as pd
from tqdm.contrib.concurrent import process_map


def print_duration(val, idx):
    if val > 1.0:
        print(f"Total duration for {idx} is {val:.1f} hours")
    elif val > 0.1:
        print(f"Total duration for {idx} is {val:.2f} hours")
    else:
        val_min = val * 60
        print(f"Total duration for {idx} is {val_min:.0f} minutes")


def read_stats(metadata_file, special_check=None):
    extra_cols = []

    df = pd.read_csv(metadata_file)
    df["duration_seconds"] = df["frames"] / df["sampling_rate"]

    if special_check == "es-ar":
        df["accent"] = df["audio_path"].apply(
            lambda x: "es-es" if "esw" in x else "es-ar"
        )
        extra_cols.append("accent")

    if special_check == "british":
        df["accent"] = df["audio_path"].apply(lambda x: os.path.basename(x)[:2])
        extra_cols.append("accent")

    stats = df.value_counts(extra_cols + ["sampling_rate", "format", "subtype"])

    if len(extra_cols) > 0:
        dur = df.groupby(extra_cols)["duration_seconds"].sum() / 3600

        for idx, val in dur.items():
            print_duration(val, idx)
    else:
        dur = df["duration_seconds"].sum() / 3600
        print_duration(dur, "all")

    print(stats)

    if special_check == "hui":
        from .datasets.hui import GENDERS

        dfg = pd.DataFrame.from_dict(GENDERS, orient="index")
        print(dfg.value_counts())


def _read_mp3_stat(file):
    try:
        from mutagen.mp3 import MP3

        audio = MP3(file)
        return audio.info.bitrate
    except Exception as e:
        print(e)
        return -1


def _read_opus_stat(file):
    # from pyogg import OpusFile
    import ffmpeg

    try:
        info = ffmpeg.probe(file)
        return int(info["format"]["bit_rate"])
    except Exception as e:
        print(e)
        return -1


def read_mp3_stats(folder):
    files = glob.glob(os.path.join(folder, "**", "*.mp3"), recursive=True)

    bit_rates = process_map(_read_mp3_stat, files, chunksize=2)

    df = pd.DataFrame({"file": files, "bit_rate": bit_rates})

    df["bit_rate_approx"] = (df["bit_rate"] / 1000).round() * 1000

    print(df["bit_rate_approx"].value_counts())

    for idx, val in df["bit_rate_approx"].value_counts().sort_index().items():
        print(idx, val)


def read_opus_stats(folder):
    files = glob.glob(os.path.join(folder, "**", "*.ogg"), recursive=True)

    bit_rates = process_map(_read_opus_stat, files, chunksize=2)

    df = pd.DataFrame({"file": files, "bit_rate": bit_rates})

    df["bit_rate_approx"] = (df["bit_rate"] / 1000).round() * 1000

    print(df["bit_rate_approx"].value_counts())

    for idx, val in df["bit_rate_approx"].value_counts().sort_index().items():
        print(idx, val)


def read_glr_stats(folder):
    df = pd.read_csv(os.path.join(folder, "manifest.csv"))

    dfg = df[["variant", "gender", "speaker_id"]].drop_duplicates()
    dfg = dfg.value_counts(["variant", "gender"]).sort_index()

    print(dfg)


if __name__ == "__main__":
    import fire

    fire.Fire()
