import ast
import glob
import math
import os
from itertools import chain

import pandas as pd
from omegaconf import DictConfig
from tqdm.contrib.concurrent import process_map

from ...const import AUDIO, AUDIO_METADATA, MANIFEST
from ..utils import (
    ffmpeg_reformat_to_buffer,
    get_file_info_as_dict,
    preface,
)


def get_segment_map(path):
    """Reads a LibriSpeech segment file into a DataFrame."""
    df = pd.read_csv(path, sep=" ", names=["segment", "start_time", "end_time"])

    return df


def get_utterance_map(path):
    """Reads a LibriSpeech utterance map file into a DataFrame."""
    df = pd.read_csv(path, sep=" ", names=["filename", "segment"], skiprows=2)

    with open(path, "r") as f:
        is_sentences_only = f.readline()

    is_sentences_only = ast.literal_eval(is_sentences_only.rstrip().split(":")[-1])

    return df, is_sentences_only


def get_time_range(speaker_id: str, chapter_id: str, cfg: DictConfig):
    """Gets the start and end times for all utterances in a chapter."""
    utterance_map_path = os.path.join(
        cfg.data.raw_data_root, cfg.data.mp3_root, speaker_id, "utterance_map.txt"
    )

    utterance_map, is_sentences_only = get_utterance_map(utterance_map_path)
    ext = "sents.seg" if is_sentences_only else "seg"

    segment_map_path = os.path.join(
        cfg.data.raw_data_root,
        cfg.data.mp3_root,
        speaker_id,
        chapter_id,
        f"{speaker_id}-{chapter_id}.{ext}.txt",
    )

    segment_map = get_segment_map(segment_map_path)

    time_range = utterance_map.merge(segment_map, on="segment").set_index("filename")[
        ["start_time", "end_time"]
    ]

    return time_range


def get_hq_mp3_and_copy(chapter_path: str, split: str, cfg: DictConfig) -> None:
    """
    Processes a single LibriSpeech-HQ chapter.

    This function reads the source MP3, slices it into individual utterances
    based on the segment maps, and saves each utterance as a separate WAV file.
    """
    sampling_rate = cfg.audio.sampling_rate
    bit_depth = cfg.audio.bit_depth
    num_channels = cfg.audio.channels

    chapter_id = os.path.basename(chapter_path)
    speaker_id = os.path.basename(os.path.dirname(chapter_path))

    mp3_path = os.path.join(
        cfg.data.raw_data_root,
        cfg.data.mp3_root,
        speaker_id,
        chapter_id,
        f"{chapter_id}.mp3",
    )

    if not os.path.exists(mp3_path):
        raise ValueError(f"MP3 file does not exist: {mp3_path}")

        return [
            {
                "file": mp3_path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
                "error": "MP3 file does not exist.",
            }
        ]

    file_info = get_file_info_as_dict(mp3_path)

    audio = ffmpeg_reformat_to_buffer(
        mp3_path,
        sampling_rate,
        num_channels,
    )

    time_range = get_time_range(speaker_id, chapter_id, cfg)

    metadata = []

    segment_paths = []
    segments = []

    for filename, row in time_range.iterrows():
        start_time = row["start_time"]
        end_time = row["end_time"]

        segment_audio = audio[
            math.floor(start_time * sampling_rate) : math.ceil(end_time * sampling_rate)
        ]

        segment_path = os.path.join(
            cfg.data.cleaned_data_root,
            cfg.dataset.name,
            AUDIO,
            cfg.dataset.subset,
            split,
            speaker_id,
            chapter_id,
            f"{filename}.wav",
        )

        os.makedirs(os.path.dirname(segment_path), exist_ok=True)

        segments.append(segment_audio)
        segment_paths.append(segment_path)

        m = {
            "file": mp3_path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
            "cleaned_path": segment_path.replace(
                cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
            ),
            "start_time": start_time,
            "end_time": end_time,
            **file_info,
        }

        metadata.append(m)

    process_map(soundfile_export, segments, segment_paths, [cfg] * len(segments))

    return metadata


def organize_split(split: str, cfg: DictConfig) -> None:
    """Organizes all chapters for a single LibriSpeech-HQ dataset split."""
    raw_split = cfg.splits[split]

    raw_data_root = cfg.data.raw_data_root

    src_info = []

    for source in raw_split:
        src_glob = source.glob
        recursive = source.recursive

        chapters = glob.glob(os.path.join(raw_data_root, src_glob), recursive=recursive)

        info = process_map(
            get_hq_mp3_and_copy,
            chapters,
            [split] * len(chapters),
            [cfg] * len(chapters),
            chunksize=1,
        )

        src_info += list(chain(*info))

    df = pd.DataFrame(src_info)

    audio_metadata_path = os.path.join(
        cfg.data.cleaned_data_root,
        cfg.dataset.name,
        AUDIO_METADATA,
        cfg.dataset.subset,
        f"{split}.csv",
    )

    os.makedirs(os.path.dirname(audio_metadata_path), exist_ok=True)

    df.to_csv(audio_metadata_path, index=False)

    # print(df)

    manifest = df[["file", "cleaned_path", "start_time", "end_time"]].copy()
    manifest["file"] = manifest["file"].replace("$CLEANED_DATA_ROOT/", "")
    manifest["subset"] = cfg.dataset.subset

    manifest["dsid_chapter_id"] = manifest["file"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )
    manifest["dsid_speaker_id"] = manifest["file"].apply(
        lambda x: os.path.basename(os.path.dirname(os.path.dirname(x)))
    )
    manifest["langcode"] = "eng"
    manifest["language"] = "English"

    manifest["license"] = "CC-BY-4.0; Public Domain"

    print(manifest)

    manifest_path = os.path.join(
        cfg.data.cleaned_data_root,
        cfg.dataset.name,
        MANIFEST,
        cfg.dataset.subset,
        f"{split}.csv",
    )

    print(manifest_path)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    manifest.to_csv(manifest_path, index=False)


def organize_librispeech_hq(cfg: DictConfig) -> None:
    """Main entry point for organizing the LibriSpeech-HQ dataset."""
    preface(cfg)

    for split in cfg.splits:
        print(f"Organizing split: {split}")
        organize_split(split=split, cfg=cfg)
