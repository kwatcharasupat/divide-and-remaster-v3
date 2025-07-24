import os
from typing import List

import pandas as pd
from omegaconf import DictConfig
from tqdm.contrib.concurrent import process_map

from ...const import AUDIO, MANIFEST
from ..utils import (
    preface,
)
from ..utils.ffmpeg import (
    ffmpeg_convert_audio_with_downmix,
    ffmpeg_mix_audio,
)

CORRECTION_DICT = {
    "Jokers Jacks & Kings - Sea Of Leaves": "Jokers, Jacks & Kings - Sea Of Leaves",
    "Patrick Talbot - Set Free Me": "Patrick Talbot - Set Me Free",
}

import json
import tempfile
from collections import defaultdict


def _handle_single_track_stem(
    stem_src_path: str,
    src_path: str,
    dst_path: str,
    genre: str,
    subset: str,
    split: str,
    cfg: DictConfig,
):
    out = ffmpeg_convert_audio_with_downmix(
        stem_src_path, dst_path, subset, split, cfg, return_dst_names=True
    )

    if isinstance(out, dict):
        return out
    elif isinstance(out, list):
        return [
            {
                "file": src_path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
                "cleaned_path": dst_path.replace(
                    cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
                ),
                "subset": subset,
                "genre": genre,
            }
            for dst_path in out
        ]

    else:
        raise ValueError(f"Unsupported output type: {type(out)}")


def _handle_multi_track_stem(
    stem_src_paths: List[str],
    src_path: str,
    dst_path: str,
    genre: str,
    subset: str,
    split: str,
    cfg: DictConfig,
):
    downmix_mode = cfg.audio.get("downmix_mode", "auto")
    if downmix_mode == "auto":
        out = ffmpeg_mix_audio(stem_src_paths, dst_path, cfg)

        if out is not None:
            return out

        ret = {
            "file": src_path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
            "cleaned_path": dst_path.replace(
                cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
            ),
            "subset": subset,
            "genre": genre,
        }

        return [ret]

    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            dst_path_tmp = dst_path.replace(cfg.data.cleaned_data_root, tmpdir)

            ffmpeg_mix_audio(stem_src_paths, dst_path_tmp, cfg, ac=2)

            out = ffmpeg_convert_audio_with_downmix(
                dst_path_tmp, dst_path, subset, split, cfg, return_dst_names=True
            )

            if isinstance(out, dict):
                return out
            elif isinstance(out, list):
                return [
                    {
                        "file": src_path.replace(
                            cfg.data.raw_data_root, "$RAW_DATA_ROOT"
                        ),
                        "cleaned_path": dst_path.replace(
                            cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
                        ),
                        "subset": subset,
                        "genre": genre,
                    }
                    for dst_path in out
                ]
            else:
                raise ValueError(f"Unsupported output type: {type(out)}")


def organize_file(
    file: str,
    split: str,
    cfg: DictConfig,
):
    rets = []

    subset = cfg.dataset.subset

    src_path = os.path.join(cfg.data.raw_data_root, cfg.path.data_root, file)

    track_manifest_path = os.path.join(src_path, "data.json")

    with open(track_manifest_path) as f:
        track_manifest = json.load(f)

    genre = track_manifest["genre"]

    stems = defaultdict(list)

    for stem in track_manifest["stems"]:
        for component in stem["tracks"]:
            if not component["has_bleed"]:
                stems[stem["stemName"]].append(component["id"])

    downmix_mode = cfg.audio.get("downmix_mode", "auto")

    for stem, components in stems.items():
        stem_src_paths = [
            os.path.join(src_path, stem, f"{component}.wav") for component in components
        ]

        filename = os.path.basename(src_path)

        subset = cfg.dataset.subset.format(stem_combination=stem)

        dst_path = os.path.join(
            cfg.data.cleaned_data_root,
            cfg.dataset.name,
            AUDIO,
            subset,
            split,
            f"{filename}.wav",
        )

        if len(stem_src_paths) == 1:
            out = _handle_single_track_stem(
                stem_src_paths[0], src_path, dst_path, genre, subset, split, cfg
            )
        else:
            out = _handle_multi_track_stem(
                stem_src_paths, src_path, dst_path, genre, subset, split, cfg
            )

        if isinstance(out, dict):
            return out
        elif isinstance(out, list):
            rets += out
        else:
            raise ValueError(f"Unsupported output type: {type(out)}")

    if downmix_mode == "auto":
        channels = [""]
    elif downmix_mode in ["lr", "split_lr"]:
        channels = ["_left", "_right"]
    elif downmix_mode == "split_all":
        channels = [f"_ch{i}" for i in range(2)]
    else:
        raise ValueError(f"Unsupported downmix mode: {downmix_mode}")

    for combi, stems_ in cfg.stem_combinations.items():
        if combi in cfg.stems:
            continue

        stems_ = [stem for stem in stems_ if stem in stems]

        for channel in channels:
            converted_src_paths = [
                os.path.join(
                    cfg.data.cleaned_data_root,
                    cfg.dataset.name,
                    AUDIO,
                    cfg.dataset.subset.format(stem_combination=stem),
                    split,
                    f"{filename}{channel}.wav",
                )
                for stem in stems_
            ]

            subset = cfg.dataset.subset.format(stem_combination=combi)

            dst_path = os.path.join(
                cfg.data.cleaned_data_root,
                cfg.dataset.name,
                AUDIO,
                subset,
                split,
                f"{filename}{channel}.wav",
            )

            ffmpeg_mix_audio(converted_src_paths, dst_path, cfg)

            ret = {
                "file": src_path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
                "cleaned_path": dst_path.replace(
                    cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
                ),
                "subset": subset,
                "genre": genre,
            }

            rets.append(ret)

    return rets


def organize_split(manifest, split, cfg):
    files = manifest["file"].tolist()

    metadata = process_map(
        organize_file,
        files,
        [split] * len(files),
        [cfg] * len(files),
        chunksize=1,
    )

    metadata = [item for sublist in metadata for item in sublist if item is not None]
    manifest = pd.DataFrame(metadata)

    manifest["split"] = split
    manifest["license"] = "NC-RCL"

    for subset, dfg in manifest.groupby("subset"):
        manifest_path = os.path.join(
            cfg.data.cleaned_data_root,
            cfg.dataset.name,
            MANIFEST,
            subset,
            f"{split}.csv",
        )

        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

        dfg.to_csv(manifest_path, index=False)


def get_split(fold: int, cfg) -> str:
    for split in cfg.splits:
        if fold in cfg.splits[split].folds:
            return split

    raise ValueError(f"Invalid fold: {fold}")


def make_manifest_and_splits(cfg: DictConfig) -> None:
    split_df = pd.read_csv(os.path.join(cfg.data.raw_data_root, cfg.path.splits))

    split_df = split_df.rename(columns={"split": "fold", "song_id": "file"})

    split_df["split"] = split_df["fold"].apply(lambda x: get_split(x, cfg))

    split_df = split_df.drop(columns=["fold"])

    split_df.to_csv(
        os.path.join(cfg.data.raw_data_root, cfg.path.manifest),
        index=False,
    )


def organize_moisesdb(cfg: DictConfig) -> None:
    preface(cfg)

    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    if not os.path.exists(manifest_path):
        make_manifest_and_splits(cfg)

    manifest = pd.read_csv(manifest_path)

    for split in cfg.splits:
        print()
        print("".join(["="] * 80))
        print(f"Organizing split: {split}")
        print("".join(["="] * 80))
        print()

        organize_split(
            manifest=manifest[manifest["split"] == split], split=split, cfg=cfg
        )
