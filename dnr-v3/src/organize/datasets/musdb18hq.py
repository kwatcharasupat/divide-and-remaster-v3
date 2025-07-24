import glob
import os
from pprint import pprint

import ffmpeg
import pandas as pd
from omegaconf import DictConfig
from tqdm.contrib.concurrent import process_map

from ...const import AUDIO, MANIFEST
from ..utils import (
    preface,
)
from ..utils.ffmpeg import ffmpeg_convert_audio_with_downmix, ffmpeg_get_acodec

CORRECTION_DICT = {
    "Jokers Jacks & Kings - Sea Of Leaves": "Jokers, Jacks & Kings - Sea Of Leaves",
    "Patrick Talbot - Set Free Me": "Patrick Talbot - Set Me Free",
}


def organize_file(
    file: str,
    genre: str,
    license_: str,
    split: str,
    cfg: DictConfig,
):
    subset = cfg.dataset.subset

    src_path = os.path.join(
        cfg.data.raw_data_root, cfg.splits[split].glob.replace("*", file)
    )

    rets = []

    for stem in cfg.stems:
        stem_src_path = os.path.join(src_path, f"{stem}.wav")
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

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        out = ffmpeg_convert_audio_with_downmix(
            stem_src_path, dst_path, subset, split, cfg, return_dst_names=True
        )

        if isinstance(out, dict):
            return out
        elif isinstance(out, list):
            for dst_path in out:
                ret = {
                    "file": src_path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
                    "cleaned_path": dst_path.replace(
                        cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
                    ),
                    "subset": subset,
                    "genre": genre,
                    "license": license_,
                }

                rets.append(ret)
        else:
            raise ValueError(f"Unsupported output type: {type(out)}")

    downmix_mode = cfg.audio.get("downmix_mode", "auto")

    if downmix_mode == "auto":
        channels = [""]
    elif downmix_mode in ["lr", "split_lr"]:
        channels = ["_left", "_right"]
    elif downmix_mode == "split_all":
        channels = [f"_ch{i}" for i in range(2)]
    else:
        raise ValueError(f"Unsupported downmix mode: {downmix_mode}")

    for combi, stems in cfg.stem_combinations.items():
        if combi in cfg.stems:
            continue

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
                for stem in stems
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

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            acodec = ffmpeg_get_acodec(cfg.audio.bit_depth)

            input_streams = [ffmpeg.input(src_path) for src_path in converted_src_paths]

            ffmpeg.filter(
                input_streams,
                "amix",
                inputs=len(input_streams),
                dropout_transition=0,
                duration="longest",
            ).output(
                dst_path,
                ac=cfg.audio.channels,
                ar=cfg.audio.sampling_rate,
                acodec=acodec,
                loglevel="error",
            ).run(overwrite_output=True)

            ret = {
                "file": src_path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
                "cleaned_path": dst_path.replace(
                    cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
                ),
                "subset": subset,
                "genre": genre,
                "license": license_,
            }

            rets.append(ret)

    return rets


def organize_split(manifest, split, cfg):
    files = manifest["file"].tolist()
    genres = manifest["genre"].tolist()
    licenses = manifest["license"].tolist()

    metadata = process_map(
        organize_file,
        files,
        genres,
        licenses,
        [split] * len(files),
        [cfg] * len(files),
        chunksize=1,
    )

    metadata = [ret for rets in metadata for ret in rets if ret is not None]
    manifest = pd.DataFrame(metadata)

    manifest["split"] = split

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


def get_license(track_name, train_list, val_list, test_list):
    if track_name in val_list:
        return "val"

    if track_name in test_list:
        return "test"

    if track_name in train_list:
        return "train"

    pprint(sorted(train_list))

    raise ValueError(f"Track `{track_name}` not found in any split list")


def make_manifest_and_splits(cfg: DictConfig) -> None:
    trainval_tracks = glob.glob(
        os.path.join(cfg.data.raw_data_root, cfg.splits.train.glob)
    )
    val_tracks = cfg.splits.val.include

    train_tracks = [
        os.path.basename(track)
        for track in trainval_tracks
        if os.path.basename(track) not in val_tracks
    ]
    test_tracks = glob.glob(os.path.join(cfg.data.raw_data_root, cfg.splits.test.glob))
    test_tracks = [os.path.basename(track) for track in test_tracks]

    assert len(train_tracks) == 86
    assert len(val_tracks) == 14
    assert len(test_tracks) == 50

    license_path = cfg.path.licenses

    license_df = pd.read_csv(os.path.join(cfg.data.raw_data_root, license_path))

    print(f"Found {len(license_df)} licenses")

    license_df["Track Name"] = license_df["Track Name"].apply(
        lambda x: CORRECTION_DICT.get(x, x)
    )

    license_df["naive_split"] = license_df["Track Name"].apply(
        lambda x: get_license(x, train_tracks, val_tracks, test_tracks)
    )

    license_df["is_blocked"] = license_df["Track Name"].apply(
        lambda x: x in cfg.blocklist
    )

    license_df["split"] = license_df["naive_split"] + license_df["is_blocked"].apply(
        lambda x: "_blocked" if x else ""
    )

    assert len(license_df[license_df["is_blocked"]]) == len(cfg.blocklist)

    license_df = license_df.rename(
        columns={
            "Track Name": "file",
            "Genre": "genre",
            "Source": "source",
            "License": "license",
        }
    )

    license_df = license_df[["file", "genre", "source", "license", "split"]]

    license_df.to_csv(
        os.path.join(cfg.data.raw_data_root, cfg.path.manifest), index=False
    )


def organize_musdb18hq(cfg: DictConfig) -> None:
    preface(cfg)

    manifest_path = os.path.join(cfg.data.raw_data_root, cfg.path.manifest)

    # if not os.path.exists(manifest_path):
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
