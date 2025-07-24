import glob
import os

import librosa
import pandas as pd
from tqdm.contrib.concurrent import process_map

from ...const import AUDIO, MANIFEST
from ..utils import ffmpeg_reformat_to_buffer, soundfile_export


def get_data_list(data_source, cfg):
    glob_path = os.path.join(cfg.data.raw_data_root, data_source.glob)

    print(f"Searching for files in {glob_path}")

    data_list = glob.glob(glob_path, recursive=False)

    print(f"Found {len(data_list)} files for {data_source}")
    # print(data_list[:5])

    df = pd.DataFrame(data_list, columns=["file"])
    df["gender"] = data_source.gender
    df["language"] = data_source.language
    df["set_"] = data_source["set"]
    return df


def make_manifest_and_splits(cfg):
    data_sources = cfg.data.sources

    manifests = process_map(get_data_list, data_sources, [cfg] * len(data_sources))

    manifest = pd.concat(manifests)

    manifest["split"] = "train"

    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    manifest.to_csv(manifest_path, index=False)


def organize_file(file, gender, langcode, set_, split, cfg):
    subset = cfg.dataset.subset

    filename = os.path.basename(file)

    file = os.path.expandvars(file)

    output_path = os.path.join(
        cfg.data.cleaned_data_root,
        cfg.dataset.name,
        AUDIO,
        subset,
        split,
        f"{set_}_{filename}",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    audio = ffmpeg_reformat_to_buffer(
        file, cfg.audio.sampling_rate, cfg.audio.channels
    ).T

    audio, (start_index, end_index) = librosa.effects.trim(
        audio,
        top_db=40,
        frame_length=int(cfg.audio.sampling_rate * 40 / 1000),
        hop_length=int(cfg.audio.sampling_rate * 10 / 1000),
    )

    audio = audio[:, start_index:end_index]

    soundfile_export(audio.T, output_path, cfg)

    metadata = {
        "file": file.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
        "cleaned_path": output_path.replace(
            cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
        ),
        "speaker_gender": "male",
        "langcode": "zlm",
        "language": "Malay",
        "dsid_set": set_,
        "subset": subset,
        "split": split,
    }

    return metadata


def organize_split(manifest, split, cfg):
    files = manifest["file"].tolist()
    genders = manifest["gender"].apply(lambda x: x[0]).tolist()
    languages = manifest["language"].tolist()
    sets_ = manifest["set_"].tolist()

    metadata = process_map(
        organize_file,
        files,
        genders,
        languages,
        sets_,
        [split] * len(files),
        [cfg] * len(files),
        chunksize=1,
    )

    manifest = pd.DataFrame(metadata)

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


def organize_malaya(cfg):
    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    # if not os.path.exists(manifest_path):
    make_manifest_and_splits(cfg)

    manifest = pd.read_csv(manifest_path)

    for split in manifest["split"].unique():
        split_manifest = manifest[manifest["split"] == split]
        organize_split(split_manifest, split, cfg)
