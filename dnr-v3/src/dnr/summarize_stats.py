import glob
import os

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def read_audio_metadata_file(file):
    df = pd.read_csv(file)
    df.columns = ["stem"] + df.columns[1:].tolist()
    df = pd.melt(df, id_vars=["stem"])
    df["file"] = os.path.basename(file).split(".")[0]
    return df


def summarize_audio_metadata(
    root="/fsx_vfx/projects/aa-cass/data/dnr-variants/dnr-v3-com-48k-smad-multi-v2/audio_metadata",
):
    dfs_list = []

    for variant in os.listdir(root):
        for split in os.listdir(os.path.join(root, variant)):
            files = glob.glob(os.path.join(root, variant, split, "*.csv"))
            dfs = process_map(read_audio_metadata_file, files, chunksize=1)

            df = pd.concat(dfs)
            df["variant"] = variant
            df["split"] = split

            dfs_list.append(df)

    df = pd.concat(dfs_list)

    df.to_csv("audio_metadata.csv", index=False)


def read_manifest_file(file):
    df = pd.read_csv(file)

    paths = file.split("/")

    df["variant"] = paths[-4]
    df["split"] = paths[-3]
    df["file"] = paths[-2]
    df["stem"] = paths[-1].split(".")[0]

    return df


def summarize_manifest(
    root="/fsx_vfx/projects/aa-cass/data/dnr-variants/dnr-v3-com-48k-smad-multi-v2/manifest",
):
    TEST_LANGS = [
        "multi-v2b",
        "eng",
        "deu",
        "fao",
        "fra",
        "spa",
        "cmn",
        "eus",
        "jpn",
        "inc",
        "dra",
        "bnt",
        "yor",
    ]

    files = []

    for lang in tqdm(TEST_LANGS):
        files += glob.glob(os.path.join(root, lang, "**", "*.csv"), recursive=True)

    dfs = process_map(read_manifest_file, files, chunksize=1)

    df = pd.concat(dfs)

    df.to_csv("manifest.csv", index=False)


if __name__ == "__main__":
    import fire

    fire.Fire()
