import glob
import os

import pandas as pd
from tqdm.contrib.concurrent import process_map


def clean_manifest(file):
    df = pd.read_csv(file)

    df["file"] = df["file"].str.replace(
        "/fsx_vfx/projects/aa-cass/data/cass-data-cleaned/", ""
    )

    cols = {
        "file": "raw_path",
        "start_sample": "start_time_sample",
        "length_sample": "segment_duration_sample",
        "segment_start_sample": "segment_start_offset_sample",
        "lufs": "event_premaster_lufs",
        "submix_lufs": "track_premaster_lufs",
    }

    df = df.rename(columns=cols)

    df = df[cols.values()]

    file = file.replace("manifest", "cleaned_manifest")

    os.makedirs(os.path.dirname(file), exist_ok=True)

    df.to_csv(file, index=False)


def clean_manifests(root="/root/data/cass-data/dnr-v3/dnr-v3"):
    for variant in os.listdir(root):
        if os.path.isfile(f"{root}/{variant}"):
            continue
        for split in os.listdir(f"{root}/{variant}/manifest"):
            print(f"{root}/{variant}/manifest/{split}")
            files = glob.glob(
                f"{root}/{variant}/manifest/{split}/**/*.csv", recursive=True
            )
            process_map(clean_manifest, files, chunksize=8)


if __name__ == "__main__":
    import fire

    fire.Fire(clean_manifests)
