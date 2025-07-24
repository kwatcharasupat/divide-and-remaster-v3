import glob
import os

import ffmpeg
import pandas as pd
from tqdm.contrib.concurrent import process_map

from ...const import AUDIO, MANIFEST
from ..utils import ffmpeg_get_acodec


def file_name_to_speaker_track(file_name):
    out = os.path.basename(file_name).split(".")[0].split("_")[-2:]

    if len(out) < 2:
        print(f"Error: {file_name}")
        raise ValueError("Invalid file name")

    speaker, track = out

    return speaker, track


def get_data_list(data_source, cfg):
    glob_path = os.path.join(cfg.data.raw_data_root, data_source.glob)

    print(f"Searching for files in {glob_path}")

    data_list = glob.glob(glob_path, recursive=True)

    print(f"Found {len(data_list)} files for {data_source}")

    df = pd.DataFrame(data_list, columns=["file"])
    df["gender"] = data_source.gender
    df["speaker_id"] = data_source.speaker_id

    df["split"] = data_source.split

    print(data_source)

    return df


def make_manifest_and_splits(cfg):
    data_sources = cfg.data.sources

    manifests = process_map(get_data_list, data_sources, [cfg] * len(data_sources))

    manifest = pd.concat(manifests)

    print(manifest)

    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    manifest.to_csv(manifest_path, index=False)


def organize_file(
    file,
    gender,
    # variant,
    speaker_id,
    #   track_id,
    split,
    cfg,
):
    subset = cfg.dataset.subset

    filename = os.path.basename(file)

    file = os.path.expandvars(file)

    filename_wav = os.path.splitext(filename)[0] + ".wav"

    output_path = os.path.join(
        cfg.data.cleaned_data_root, cfg.dataset.name, AUDIO, subset, split, filename_wav
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    acodec = ffmpeg_get_acodec(cfg.audio.bit_depth)

    try:
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
    except Exception as e:
        print(f"Error: {file}")
        print(e)

        if os.path.exists(output_path):
            os.remove(output_path)

        return None

    metadata = {
        "file": file.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
        "cleaned_path": output_path.replace(
            cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
        ),
        "speaker_gender": gender,
        "langcode": "jpn",
        "language": "Japanese",
        "subset": subset,
        "dsid_speaker_id": speaker_id,
        # "dsid_track_id": track_id,
        # "dsid_variant": variant,
        "split": split,
    }

    return metadata


def organize_split(manifest, split, cfg):
    files = manifest["file"].tolist()
    genders = manifest["gender"].apply(lambda x: x[0]).tolist()
    # languages = manifest["language"].tolist()
    # variants = manifest["variant"].tolist()
    speaker_ids = manifest["speaker_id"].tolist()
    # track_ids = manifest["track_id"].tolist()

    metadata = process_map(
        organize_file,
        files,
        genders,
        # languages,
        # variants,
        speaker_ids,
        # track_ids,
        [split] * len(files),
        [cfg] * len(files),
        chunksize=1,
    )

    metadata = [m for m in metadata if m is not None]

    manifest = pd.DataFrame(metadata)

    manifest["license"] = "Apache 2.0"

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


def organize_ukrainian(cfg):
    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    # if not os.path.exists(manifest_path):
    make_manifest_and_splits(cfg)

    manifest = pd.read_csv(manifest_path)

    splits = manifest["split"].unique()

    print("This dataset only has : ", splits)

    for split in splits:
        split_manifest = manifest[manifest["split"] == split]
        organize_split(split_manifest, split, cfg)
