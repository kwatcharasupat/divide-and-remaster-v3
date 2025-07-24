import os

import pandas as pd
from tqdm.contrib.concurrent import process_map

from ...const import MANIFEST
from ..utils import get_data_list, preface
from ..utils.prototypes import _generic_organize_file

ASC_LICENSE = "CC-BY-4.0"
ASC_LANGCODE = "apc"
ASC_LANGNAME = "Arabic, Levantine"
ASC_SPEAKER_GENDER = "male"
ASC_SPLIT = "train"


def make_manifest_and_splits(cfg):
    data_sources = cfg.data.sources

    manifests = process_map(get_data_list, data_sources, [cfg] * len(data_sources))
    manifest = pd.concat(manifests)
    manifest["split"] = ASC_SPLIT

    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    manifest.to_csv(manifest_path, index=False)


def organize_file(file, split, cfg):
    subset = cfg.dataset.subset

    return _generic_organize_file(
        src_path=file,
        split=split,
        subset=subset,
        cfg=cfg,
        langcode=ASC_LANGCODE,
        langname=ASC_LANGNAME,
        speaker_gender=ASC_SPEAKER_GENDER,
    )


def organize_split(manifest, split, cfg):
    files = manifest["file"].tolist()

    metadata = process_map(
        organize_file,
        files,
        [split] * len(files),
        [cfg] * len(files),
        chunksize=1,
    )

    manifest = pd.DataFrame(metadata)

    manifest["license"] = ASC_LICENSE
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


def organize_arabic_speech_corpus(cfg):
    preface(cfg)

    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    if (
        not os.path.exists(manifest_path)
        or cfg.get("remake_splits", False)
        or cfg.get("remake_manifest", False)
    ):
        make_manifest_and_splits(cfg)

    manifest = pd.read_csv(manifest_path)

    for split in manifest["split"].unique():
        split_manifest = manifest[manifest["split"] == split]
        organize_split(split_manifest, split, cfg)
