import glob
import os
from itertools import chain

import pandas as pd
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
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
    return df


def make_manifest_and_splits(cfg):
    data_sources = cfg.data.sources

    manifests = process_map(get_data_list, data_sources, [cfg] * len(data_sources))

    manifest = pd.concat(manifests)

    manifest["split"] = "train"

    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    manifest.to_csv(manifest_path, index=False)


def organize_file(file, split, cfg):
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

    audio_segment = AudioSegment.from_file(file, format="wav")

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
                "speaker_gender": "male",
                "langcode": "por",
                "language": "Portuguese",
                "subset": subset,
                "split": split,
            }
        )

        segment_idx += 1

    return manifest


def organize_split(manifest, split, cfg):
    files = manifest["file"].tolist()

    metadata = process_map(
        organize_file,
        files,
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


def organize_portuguese(cfg):
    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    # if not os.path.exists(manifest_path):
    make_manifest_and_splits(cfg)

    manifest = pd.read_csv(manifest_path)

    for split in manifest["split"].unique():
        split_manifest = manifest[manifest["split"] == split]
        organize_split(split_manifest, split, cfg)
