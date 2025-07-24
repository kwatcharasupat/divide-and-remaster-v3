import os
import shutil

import torch
from aa_smad_cli.predictor import crnn_predictor as smad_predictor
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

tqdm.pandas()

import pandas as pd
import soundfile as sf


def get_duration(path):
    duration = sf.info(path).frames / sf.info(path).samplerate

    return {"cleaned_path": path, "duration": duration}


def copy_(path, src_replace, dst_replace):
    out_path = path.replace(src_replace, dst_replace)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copy(path, out_path)


@torch.inference_mode()
def check_file_no_vox(
    manifest="/fsx_vfx/projects/aa-cass/data/cass-data-cleaned/effects-fsd50k/manifest/full-v3/com/effects-no-vox/48k/{split}.csv",
    src_replace="full-v3",
    dst_replace="full-v3-smad",
):
    smad = smad_predictor(cuda=False)
    for split in ["train", "val", "test"]:
        usable_rows = []

        df = pd.read_csv(manifest.format(split=split))

        df["cleaned_path"] = df["cleaned_path"].apply(os.path.expandvars)

        durations = process_map(get_duration, df["cleaned_path"].tolist())
        durations = pd.DataFrame(durations)

        df = df.merge(durations, on="path")

        short_df = df[df["duration"] < 1]

        process_map(
            copy_,
            short_df["cleaned_path"].tolist(),
            [src_replace] * len(short_df),
            [dst_replace] * len(short_df),
        )

        df = df[df["duration"] >= 1]

        n_paths = len(df)
        total_skipped = 0
        pc_skipped = 0
        pc_skipped_so_far = 0

        pbar = tqdm(total=n_paths)

        for i, row in df.iterrows():
            pbar.update(1)

            path = row["cleaned_path"]

            try:
                activations = smad.prediction(path)

                if (activations > 0.5).max(axis=0).mean() > 0.5:
                    # print(f"Skipping {path}: contains vox/music")
                    total_skipped += 1
                    pc_skipped = total_skipped / n_paths * 100
                    pc_skipped_so_far = total_skipped / (pbar.n) * 100

                    pbar.set_postfix(
                        {
                            "skip%": f"{pc_skipped:.2f}%",
                            "skip% so far": f"{pc_skipped_so_far:.2f}%",
                        }
                    )

                    continue

            except Exception as e:
                print(e)
                continue

            out_path = path.replace(src_replace, dst_replace)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            usable_rows.append(row)

            shutil.copy(path, out_path)

        pbar.close()

        df = pd.DataFrame(usable_rows)
        df = pd.concat([df, short_df])

        csv_path = manifest.format(split=split).replace(src_replace, dst_replace)

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        df.to_csv(csv_path, index=False)


def check_duration(
    manifest="/fsx_vfx/projects/aa-cass/data/cass-data-cleaned/effects-fsd50k/manifest/full-v3-smad/com/effects-no-vox/48k/{split}.csv",
):
    for split in ["train", "val", "test"]:
        df = pd.read_csv(manifest.format(split=split))

        durations = process_map(get_duration, df["cleaned_path"].tolist())

        durations = pd.DataFrame(durations)
        durations["duration_recomputed"] = durations["duration"]
        durations = durations.drop("duration", axis=1)

        df = df.merge(durations, on="cleaned_path")

        diff = (df["duration"] - df["duration_recomputed"]).abs()

        print(diff.describe())


if __name__ == "__main__":
    import fire

    fire.Fire()
