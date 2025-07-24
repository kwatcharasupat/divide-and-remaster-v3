import glob
import os

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map

from ...const import MANIFEST, REMAKE_MANIFEST, REMAKE_SPLITS
from ..utils import (
    preface,
)
from ..utils.prototypes import _generic_organize_file

AISHELL3_STRATIFY_GROUPS = {
    ("female", "north", "A"): "female-north-B",
    ("male", "north", "D"): "male-north-C",
    ("male", "south", "C"): "male-south-B",
}


AISHELL3_LANGCODE = "cmn"
AISHELL3_LANGNAME = "Chinese, Mandarin"
AISHELL3_LICENSE = "Apache 2.0"


def organize_file(file: str, gender: str, split: str, cfg: DictConfig):
    subset = cfg.dataset.subset
    speaker = file.split("/")[-2]

    return _generic_organize_file(
        src_path=file,
        split=split,
        subset=subset,
        cfg=cfg,
        langcode=AISHELL3_LANGCODE,
        langname=AISHELL3_LANGNAME,
        speaker_id=speaker,
    )


def organize_split(manifest: pd.DataFrame, split: str, cfg: DictConfig):
    df = manifest

    files = df["file"].tolist()
    genders = df["gender"].tolist()

    metadata = process_map(
        organize_file,
        files,
        genders,
        [split] * len(files),
        [cfg] * len(files),
        chunksize=1,
    )

    metadata = [m for m in metadata if m is not None]
    manifest = pd.DataFrame(metadata)

    manifest["langcode"] = AISHELL3_LANGCODE
    manifest["language"] = AISHELL3_LANGNAME
    manifest["license"] = AISHELL3_LICENSE

    manifest["split"] = split

    manifest_path = os.path.join(
        cfg.data.cleaned_data_root,
        cfg.dataset.name,
        MANIFEST,
        cfg.dataset.subset,
        f"{split}.csv",
    )

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    manifest.to_csv(manifest_path, index=False)


def make_manifest(cfg):
    speaker_info = pd.read_csv(
        os.path.join(cfg.data.raw_data_root, cfg.path.speaker_info),
        sep="\t",
        skiprows=3,
        names=["speaker_id", "age_group", "gender", "accent"],
    )

    dfs = []

    for split in ["train", "test"]:
        files = glob.glob(
            os.path.join(
                cfg.data.raw_data_root, "aishell3", split, "wav", "**", "*.wav"
            )
        )

        df = pd.DataFrame(files, columns=["file"])
        df["speaker_id"] = df["file"].apply(lambda x: x.split("/")[-2])
        df["original_split"] = split

        dfs.append(df)

    manifest = pd.concat(dfs)

    manifest = manifest.merge(speaker_info, on="speaker_id", how="left")

    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    manifest.to_csv(manifest_path, index=False)

    print(f"Manifest saved to {cfg.path.manifest}")
    print(manifest.head())


def make_splits(manifest, cfg):
    train_df = manifest[manifest["original_split"] == "train"].copy()

    speaker_info = pd.read_csv(
        os.path.join(cfg.data.raw_data_root, cfg.path.speaker_info),
        sep="\t",
        skiprows=3,
        names=["speaker_id", "age_group", "gender", "accent"],
    )

    speaker_info["gender"] = speaker_info["gender"].str.rstrip()
    speaker_info["accent"] = speaker_info["accent"].str.rstrip()

    train_speaker_ids = train_df["speaker_id"].unique()
    train_speaker_ids = sorted(list(train_speaker_ids))

    speaker_info = speaker_info[speaker_info["speaker_id"].isin(train_speaker_ids)]
    speaker_info = speaker_info.set_index("speaker_id").loc[train_speaker_ids]
    speaker_info = speaker_info.apply(
        lambda row: AISHELL3_STRATIFY_GROUPS.get(
            (row["gender"], row["accent"], row["age_group"]),
            "-".join([row["gender"], row["accent"], row["age_group"]]),
        ),
        axis=1,
    )

    assert len(train_speaker_ids) == len(speaker_info)

    speaker_info = speaker_info.values

    train_speaker_ids, val_speaker_ids = train_test_split(
        train_speaker_ids,
        test_size=cfg.splits.val.prop,
        random_state=cfg.seed,
        stratify=speaker_info,
    )

    train_df["split"] = train_df["speaker_id"].apply(
        lambda x: "train" if x in train_speaker_ids else "val"
    )

    test_df = manifest[manifest["original_split"] == "test"].copy()
    test_df["split"] = "test"

    df = pd.concat([train_df, test_df])

    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    df.to_csv(manifest_path, index=False)

    print(df.value_counts("split"))


def organize_aishell3(cfg):
    preface(cfg)

    if not os.path.exists(
        os.path.join(cfg.data.raw_data_root, cfg.path.manifest)
    ) or cfg.get(REMAKE_MANIFEST, False):
        print("Making manifest")
        make_manifest(cfg)

    manifest = pd.read_csv(os.path.join(cfg.data.raw_data_root, cfg.path.manifest))

    if "split" not in manifest.columns or cfg.get(REMAKE_SPLITS, False):
        print("Making splits")
        make_splits(manifest, cfg)
        manifest = pd.read_csv(os.path.join(cfg.data.raw_data_root, cfg.path.manifest))

    for split in cfg.splits:
        print(f"Organizing split: {split}")
        organize_split(
            manifest=manifest[manifest["split"] == split], split=split, cfg=cfg
        )
