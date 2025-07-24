import glob
import os
from itertools import chain

import pandas as pd
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map

from ...const import AUDIO, MANIFEST
from ..utils import ffmpeg_reformat_to_buffer, soundfile_export


def get_sampling_rate(file):
    """Gets the sampling rate of an audio file."""
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
    """Finds all audio files for a given data source and gets their sampling rates."""
    glob_path = os.path.join(cfg.data.raw_data_root, data_source.glob)

    print(f"Searching for files in {glob_path}")

    data_list = glob.glob(glob_path, recursive=False)
    data_list = [x for x in data_list if not os.path.basename(x).startswith(".")]

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
    """
    Creates a master manifest for the Faroese dataset and splits it by speaker.

    This function ensures that all utterances from a single speaker belong to the
    same split (train, validation, or test).
    """
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

    print("Train speakers: ", len(train_speaker_list))
    print(train_speaker_list[:20])

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
    """
    Organizes a single Faroese audio file.

    This function uses pydub to detect non-silent segments, splits the source
    file into multiple sub-clips based on these segments, and saves each as a
    new WAV file.
    """
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

    manifest = []

    audio_segment = AudioSegment.from_file(file, format="flac")

    dbfs = audio_segment.dBFS

    nonsilent_ranges = detect_nonsilent(
        audio_segment, min_silence_len=1000, silence_thresh=dbfs - 16
    )

    # print(nonsilent_ranges)

    if len(nonsilent_ranges) == 0:
        raise ValueError("No nonsilent ranges found")

    old_basename = os.path.basename(file).split(".")[0]

    for segment_idx, (start_ms, end_ms) in enumerate(nonsilent_ranges):
        start_sample = int((start_ms - 100) * cfg.audio.sampling_rate / 1000)
        end_sample = int((end_ms + 100) * cfg.audio.sampling_rate / 1000)

        start_sample = max(0, start_sample)
        end_sample = min(len(audio[0]), end_sample)

        # print(start_sample, end_sample)

        subsegment = audio[:, start_sample:end_sample]

        segment_path = os.path.join(
            cfg.data.cleaned_data_root,
            cfg.dataset.name,
            AUDIO,
            subset,
            split,
            f"{old_basename}_{segment_idx:05d}.wav",
        )

        soundfile_export(subsegment.T, segment_path, cfg)

        manifest.append(
            {
                "file": file.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
                "cleaned_path": segment_path.replace(
                    cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
                ),
                "speaker": speaker_id,
                "subset": subset,
                "split": split,
                "segment_idx": segment_idx,
                "start_time": start_ms / 1000,
                "end_time": end_ms / 1000,
            }
        )

        segment_idx += 1

    return manifest


def organize_split(manifest, split, cfg):
    """Organizes all files for a single dataset split."""
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

    metadata = list(chain(*metadata))

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


def organize_faroese(cfg):
    """Main entry point for organizing the Faroese dataset."""
    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    # if not os.path.exists(manifest_path):
    make_manifest_and_splits(cfg)

    manifest = pd.read_csv(manifest_path)

    for split in manifest["split"].unique():
        split_manifest = manifest[manifest["split"] == split]
        organize_split(split_manifest, split, cfg)
