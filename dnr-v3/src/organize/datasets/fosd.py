import os

import ffmpeg
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map

from ...const import AUDIO, MANIFEST
from ..utils import ffmpeg_get_acodec, preface


def process_file(file, split, cfg):
    subset = cfg.dataset.subset

    filename = os.path.join(cfg.data.raw_data_root, cfg.path.audio_root, f"{file}.mp3")

    set_ = file.split("_")[1]
    id_in_set = file.split("_")[-1]

    output_path = os.path.join(
        cfg.data.cleaned_data_root,
        cfg.dataset.name,
        AUDIO,
        subset,
        split,
        f"s{set_}f{id_in_set}.wav",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        ffmpeg.probe(filename)
    except ffmpeg.Error as e:
        print(f"Failed to probe {filename}")
        print(e)

        if os.path.exists(output_path):
            os.remove(output_path)

        return None

    acodec = ffmpeg_get_acodec(cfg.audio.bit_depth)

    (
        ffmpeg.input(filename)
        .output(
            output_path,
            format="wav",
            acodec=acodec,
            ar=cfg.audio.sampling_rate,
            ac=cfg.audio.channels,
            loglevel="error",
        )
        .run(overwrite_output=True)
    )

    metadata = {
        "file": file.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
        "cleaned_path": output_path.replace(
            cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
        ),
        "dsid_set_id": set_,
        "dsid_id_in_set": id_in_set,
        "subset": subset,
    }

    return metadata


def organize_split(manifest, split, cfg):
    df = manifest[manifest["split"] == split]

    files = df["file"].tolist()

    metadata = process_map(
        process_file,
        files,
        [split] * len(files),
        [cfg] * len(files),
        chunksize=1,
    )

    metadata = [m for m in metadata if m is not None]

    manifest = pd.DataFrame(metadata)

    manifest["langcode"] = "vie"
    manifest["language"] = "Vietnamese"

    manifest["license"] = "CC-BY-4.0"

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


def make_manifest_and_splits(cfg):
    train_metadata_path = os.path.join(
        cfg.data.raw_data_root, cfg.splits.train.metadata
    )

    df_train = pd.read_csv(
        train_metadata_path, names=["file", "trans", "trans2"], sep="|"
    )
    df_train["original_split"] = "train"

    test_metadata_path = os.path.join(cfg.data.raw_data_root, cfg.splits.test.metadata)

    df_test = pd.read_csv(
        test_metadata_path, names=["file", "trans", "trans2"], sep="|"
    )
    df_test["original_split"] = "test"

    df_train, df_val = train_test_split(
        df_train, test_size=cfg.splits.val.prop, random_state=cfg.seed
    )

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    df = pd.concat([df_train, df_val, df_test])[["file", "split"]].reset_index(
        drop=True
    )

    os.makedirs(
        os.path.dirname(os.path.join(cfg.data.raw_data_root, cfg.path.manifest)),
        exist_ok=True,
    )

    df.to_csv(os.path.join(cfg.data.raw_data_root, cfg.path.manifest), index=False)


def organize_fosd(cfg):
    preface(cfg)

    if not os.path.exists(os.path.join(cfg.data.raw_data_root, cfg.path.manifest)):
        print("Making manifest")
        make_manifest_and_splits(cfg)

    manifest = pd.read_csv(os.path.join(cfg.data.raw_data_root, cfg.path.manifest))

    for split in cfg.splits:
        print(f"Organizing split: {split}")
        organize_split(manifest=manifest, split=split, cfg=cfg)
