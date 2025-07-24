import glob
import os

import ffmpeg
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map

from ...const import AUDIO, MANIFEST
from ..utils import ffmpeg_get_acodec


def get_data_list(data_source, cfg):
    glob_path = os.path.join(cfg.data.raw_data_root, data_source.glob)

    print(f"Searching for files in {glob_path}")

    data_list = glob.glob(glob_path, recursive=False)

    print(f"Found {len(data_list)} files for {data_source}")
    # print(data_list[:5])

    df = pd.DataFrame(data_list, columns=["file"])
    df["gender"] = data_source.gender

    df["speaker_id"], df["track_id"] = zip(
        *df["file"].apply(lambda x: os.path.basename(x).split(".")[0].split("_")[-2:])
    )

    df["speaker_id"] = df.apply(lambda x: f'yor-{x["gender"]}{x["speaker_id"]}', axis=1)

    return df


def make_manifest_and_splits(cfg):
    data_sources = cfg.data.sources

    manifests = process_map(get_data_list, data_sources, [cfg] * len(data_sources))

    manifest = pd.concat(manifests)

    unique_speakers = manifest["speaker_id"].unique()

    speaker_ids = manifest[["speaker_id", "gender"]].drop_duplicates()

    print(f"Found {len(unique_speakers)} unique speakers")
    # print(speaker_ids)

    assert len(unique_speakers) == len(speaker_ids), "Speaker IDs are not unique"

    speaker_gender_variants = speaker_ids[["gender"]].values

    print("Distribution of gender")
    print(speaker_ids.value_counts("gender"))

    train_speaker_ids, valtest_speaker_ids = train_test_split(
        speaker_ids,
        test_size=0.3,
        random_state=cfg.seed,
        stratify=speaker_gender_variants,
    )

    valtest_speaker_gender_variants = valtest_speaker_ids[["gender"]].values

    val_speaker_ids, test_speaker_ids = train_test_split(
        valtest_speaker_ids,
        test_size=2 / 3,
        random_state=cfg.seed,
        stratify=valtest_speaker_gender_variants,
    )

    train_speaker_list = train_speaker_ids["speaker_id"].tolist()
    val_speaker_list = val_speaker_ids["speaker_id"].tolist()
    test_speaker_list = test_speaker_ids["speaker_id"].tolist()

    manifest["split"] = manifest["speaker_id"].apply(
        lambda x: "train"
        if x in train_speaker_list
        else "val"
        if x in val_speaker_list
        else "test"
    )

    vc = manifest.groupby(["split"]).value_counts(["gender"]).sort_index()

    print("Split distribution by file count")
    print(vc)

    manifest_speakers = (
        manifest.drop_duplicates(subset=["speaker_id"])
        .value_counts(["split", "gender"])
        .sort_index()
    )

    print("Split distribution by speaker count")
    print(manifest_speakers)

    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    manifest.to_csv(manifest_path, index=False)


def organize_file(file, gender, speaker_id, track_id, split, cfg):
    subset = cfg.dataset.subset

    filename = os.path.basename(file)

    file = os.path.expandvars(file)

    output_path = os.path.join(
        cfg.data.cleaned_data_root, cfg.dataset.name, AUDIO, subset, split, filename
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    acodec = ffmpeg_get_acodec(cfg.audio.bit_depth)

    (
        ffmpeg.input(file)
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
        "speaker_gender": gender,
        "subset": subset,
        "dsid_speaker_id": speaker_id,
        "dsid_track_id": track_id,
        "split": split,
    }

    return metadata


def organize_split(manifest, split, cfg):
    files = manifest["file"].tolist()
    genders = manifest["gender"].apply(lambda x: x[0]).tolist()
    speaker_ids = manifest["speaker_id"].tolist()
    track_ids = manifest["track_id"].tolist()

    metadata = process_map(
        organize_file,
        files,
        genders,
        speaker_ids,
        track_ids,
        [split] * len(files),
        [cfg] * len(files),
        chunksize=1,
    )

    manifest = pd.DataFrame(metadata)

    manifest["langcode"] = "yor"
    manifest["language"] = "Yoruba"

    manifest["license"] = "CC-BY-SA-4.0"

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


def organize_yoruba(cfg):
    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    # if not os.path.exists(manifest_path):
    make_manifest_and_splits(cfg)

    manifest = pd.read_csv(manifest_path)

    for split in ["train", "val", "test"]:
        split_manifest = manifest[manifest["split"] == split]
        organize_split(split_manifest, split, cfg)
