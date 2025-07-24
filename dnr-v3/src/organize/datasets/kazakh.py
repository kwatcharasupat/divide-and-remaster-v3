import glob
import os

import librosa
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map

from ...const import AUDIO, MANIFEST
from ..utils import ffmpeg_reformat_to_buffer, soundfile_export


def get_sampling_rate(file):
    try:
        info = sf.info(file)
        return {
            "file": file,
            "sampling_rate": info.samplerate,
        }
    except Exception as e:
        print(e)
        return {"file": file, "sampling_rate": -1}


def get_data_list(data_source, cfg):
    glob_path = os.path.join(cfg.data.raw_data_root, data_source.glob)

    print(f"Searching for files in {glob_path}")

    data_list = glob.glob(glob_path, recursive=False)

    print(f"Found {len(data_list)} files for {data_source}")
    # print(data_list[:5])

    df = pd.DataFrame(data_list, columns=["file"])

    df["speaker_id"] = df["file"].apply(lambda x: os.path.basename(x).split("_")[0])
    sampling_rates = process_map(get_sampling_rate, data_list, chunksize=1)
    df_sampling_rates = pd.DataFrame(sampling_rates)

    df = pd.merge(df, df_sampling_rates, on="file")

    df = df[df["sampling_rate"] >= 44100]

    print("Remaining files: ", len(df))

    return df


def make_manifest_and_splits(cfg):
    data_sources = cfg.data.sources

    manifests = process_map(get_data_list, data_sources, [cfg] * len(data_sources))

    manifest = pd.concat(manifests)

    speaker_ids = manifest[["speaker_id"]].drop_duplicates()

    train_speaker_ids, valtest_speaker_ids = train_test_split(
        speaker_ids,
        test_size=0.3,
        random_state=cfg.seed,
    )

    val_speaker_ids, test_speaker_ids = train_test_split(
        valtest_speaker_ids,
        test_size=2 / 3,
        random_state=cfg.seed,
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

    manifest_speakers = (
        manifest.drop_duplicates(subset=["speaker_id"])
        .value_counts(["split"])
        .sort_index()
    )

    print("Split distribution by speaker count")
    print(manifest_speakers)

    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    manifest.to_csv(manifest_path, index=False)


def organize_file(file, speaker_id, split, cfg):
    subset = cfg.dataset.subset

    filename = os.path.basename(file)

    file = os.path.expandvars(file)

    output_path = os.path.join(
        cfg.data.cleaned_data_root,
        cfg.dataset.name,
        AUDIO,
        subset,
        split,
        f"{filename}",
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
        "speaker_gender": "unknown",
        "speaker_id": speaker_id,
        "langcode": "kaz",
        "language": "Kazakh",
        "subset": subset,
        "split": split,
    }

    return metadata


def organize_split(manifest, split, cfg):
    files = manifest["file"].tolist()
    speaker_ids = manifest["speaker_id"].tolist()

    metadata = process_map(
        organize_file,
        files,
        speaker_ids,
        [split] * len(files),
        [cfg] * len(files),
        chunksize=1,
    )

    manifest = pd.DataFrame(metadata)

    manifest["license"] = "CC-BY-SA-3.0-US"

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


def organize_kazakh(cfg):
    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    # if not os.path.exists(manifest_path):
    make_manifest_and_splits(cfg)

    manifest = pd.read_csv(manifest_path)

    for split in manifest["split"].unique():
        split_manifest = manifest[manifest["split"] == split]
        organize_split(split_manifest, split, cfg)
