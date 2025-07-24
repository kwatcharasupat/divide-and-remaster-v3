import ast
import math
import os
from itertools import chain
from multiprocessing import Pool

import ffmpeg
import numpy as np
import pandas as pd
import torch
from aa_smad_cli.predictor import crnn_predictor as smad_predictor
from omegaconf import DictConfig
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ...const import AUDIO, MANIFEST
from ..license_utils import (
    URL_TO_LICENSE,
    license_family_to_usage,
    url_to_license,
    url_to_license_family,
)
from ..smad import (
    class_to_shorthand,
    get_segments_from_activation,
)
from ..utils import (
    convert_audio_with_downmix,
    ffmpeg_reformat_to_buffer,
    get_acodec_soundfile,
    preface,
    soundfile_export,
)


def load_track_metadata(path):
    """Loads and pre-processes the main FMA track metadata file."""
    # from https://github.com/mdeff/fma/blob/master/utils.py#L197-L224

    tracks = pd.read_csv(path, index_col=0, header=[0, 1])

    COLUMNS = [
        ("track", "tags"),
        ("album", "tags"),
        ("artist", "tags"),
        ("track", "genres"),
        ("track", "genres_all"),
    ]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [
        ("track", "date_created"),
        ("track", "date_recorded"),
        ("album", "date_created"),
        ("album", "date_released"),
        ("artist", "date_created"),
        ("artist", "active_year_begin"),
        ("artist", "active_year_end"),
    ]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ("small", "medium", "large")
    try:
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            "category", categories=SUBSETS, ordered=True
        )
    except (ValueError, TypeError):
        # the categories and ordered arguments were removed in pandas 0.25
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            pd.CategoricalDtype(categories=SUBSETS, ordered=True)
        )

    COLUMNS = [
        ("track", "genre_top"),
        ("track", "license"),
        ("album", "type"),
        ("album", "information"),
        ("artist", "bio"),
    ]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype("category")

    df = tracks.reset_index()

    df = df[
        [
            ("track_id", ""),
            ("set", "split"),
            ("set", "subset"),
        ]
    ]

    df.columns = ["id", "split", "subset"]

    return df


def load_raw_track_metadata(path):
    """Loads the raw, unprocessed FMA track metadata."""
    df = pd.read_csv(path)

    df = df[["track_id", "license_url"]].rename(
        columns={
            "track_id": "id",
            "license_url": "license_url",
        }
    )

    return df


def load_echonest_metadata(path):
    """Loads the Echonest features for FMA tracks."""
    df = pd.read_csv(path, index_col=0, header=[0, 1, 2])

    dfi = df[[("echonest", "audio_features", "instrumentalness")]]
    dfi = dfi.reset_index()
    dfi.columns = ["id", "instrumentalness"]

    return dfi


def load_metadata(split: str, cfg: DictConfig, src_cfg: DictConfig):
    """Loads, merges, and filters all required FMA metadata for a given split."""
    fma_subset = cfg.fma.subset

    metadata_path = os.path.join(cfg.data.raw_data_root, src_cfg.metadata)

    df = load_track_metadata(metadata_path)

    if fma_subset == "small":
        included_subsets = ["small"]
    elif fma_subset == "medium":
        included_subsets = ["small", "medium"]
    elif fma_subset in ["large", "full"]:
        included_subsets = ["small", "medium", "large"]
    else:
        raise ValueError(f"Unknown subset: {fma_subset}")

    df = df[
        (df.subset.isin(included_subsets)) & (df.split == cfg.split_name_map[split])
    ]

    print("Total tracks:", len(df))

    raw_metadata_path = os.path.join(cfg.data.raw_data_root, src_cfg.raw_metadata)

    dfr = load_raw_track_metadata(raw_metadata_path)
    df = df.merge(dfr, on="id", how="left")

    print("Tracks with raw metadata:", len(df))

    echonest_metadata_path = os.path.join(
        cfg.data.raw_data_root, src_cfg.echonest_metadata
    )
    dfe = load_echonest_metadata(echonest_metadata_path)

    df = df.merge(dfe, on="id", how="left")

    print("Tracks with echonest metadata:", len(df))

    df["license_url"] = df["license_url"].fillna("missing")

    missing_licenses = set(df["license_url"]) - set(URL_TO_LICENSE.keys())

    if missing_licenses:
        print("Supported non-cc licenses")
        print([(k, v) for k, v in URL_TO_LICENSE.items() if "cc-" not in v])

        print("Missing licenses:")
        print()
        for ml in sorted(missing_licenses):
            print(ml)

        raise ValueError("Missing licenses")

    df["license"] = df["license_url"].apply(url_to_license)
    df["license_family"] = df["license_url"].apply(url_to_license_family)
    df["license_usage"] = df["license_family"].apply(license_family_to_usage)

    return df


def get_smad_activations(
    src_path,
    smad,
    cfg,
    overwrite=False,
):
    """Runs Speech/Music Activity Detection (SMAD) on a source file and saves the results."""
    smad_path = src_path.replace(".mp3", "_smad.csv").replace(
        f"fma_{cfg.fma.subset}", f"fma_{cfg.fma.subset}_smad"
    )

    if not overwrite:
        if os.path.exists(smad_path):
            return

    frame_size_seconds = smad.frame_time

    with torch.inference_mode():
        activations = smad.prediction(src_path)

    binarized_activations = (activations > 0.5).astype(np.int32)

    binarized_activations = binarized_activations * np.array([2, 1])[:, None]

    activations_class = binarized_activations.sum(axis=0)

    # 0 = nothing, 1 = speech, 2 = music, 3 = speech + music

    segments = get_segments_from_activation(activations_class, frame_size_seconds)

    os.makedirs(os.path.dirname(smad_path), exist_ok=True)

    segments.to_csv(smad_path, index=False)


def organize_file_smad(
    track_id: int,
    license_: str,
    license_usage: str,
    license_family: str,
    split: str,
    raw_cfg: DictConfig,
    cfg: DictConfig,
    smad,
):
    """Organizes an FMA track by splitting it into segments based on SMAD results."""
    track_id_str = f"{track_id:06d}"
    folder_id = track_id_str[:3]
    src_path = os.path.join(
        cfg.data.raw_data_root, raw_cfg.path, folder_id, f"{track_id_str}.mp3"
    )

    # out = probe_and_check(src_path, cfg)

    # if out is not None:
    #     return out

    smad_path = src_path.replace(".mp3", "_smad.csv").replace(
        f"fma_{cfg.fma.subset}", f"fma_{cfg.fma.subset}_smad"
    )

    if not os.path.exists(smad_path):
        print(f"SMAD file not found: {smad_path}")
        return []

    segments = pd.read_csv(smad_path)

    metadata = []
    audios = []
    dst_paths = []

    downmix_mode = cfg.audio.get("downmix_mode", "mid")

    if downmix_mode == "auto":
        channels = ["mid"]
    elif downmix_mode == "lr":
        channels = ["left", "right"]
    else:
        raise ValueError(f"Unsupported downmix mode: {downmix_mode}")

    acodec = get_acodec_soundfile(cfg.audio.bit_depth)

    audio = (
        ffmpeg_reformat_to_buffer(
            file=src_path, sampling_rate=cfg.audio.sampling_rate, num_channels=2
        )
        .reshape(-1, 2)
        .T
    )

    chunk_idx = 0

    for _, row in segments.iterrows():
        class_ = row["class"]
        subset = cfg.dataset.subset.format(license=license_usage, inst=class_)

        short_class = class_to_shorthand(class_)

        start_time = row.start_time
        end_time = row.end_time

        if end_time - start_time < cfg.fma.min_duration_seconds:
            continue
        elif end_time - start_time > cfg.fma.max_duration_seconds:
            n_chunks = int(
                np.ceil((end_time - start_time) / cfg.fma.max_duration_seconds)
            )

            chunk_size = (end_time - start_time) / n_chunks

            start_times = np.arange(n_chunks) * chunk_size + start_time
            end_times = start_times + chunk_size

        else:
            start_times = [start_time]
            end_times = [end_time]

        for start_time_, end_time_ in zip(start_times, end_times):
            start_sample = int(start_time_ * cfg.audio.sampling_rate)
            end_sample = int(end_time_ * cfg.audio.sampling_rate)

            audio_chunk = audio[:, start_sample:end_sample]

            for channel in channels:
                if channel == "mid":
                    dst_path = os.path.join(
                        cfg.data.cleaned_data_root,
                        cfg.dataset.name,
                        AUDIO,
                        subset,
                        split,
                        f"{track_id_str}_{short_class}{chunk_idx:03d}_mid.wav",
                    )
                    chunk = audio_chunk.mean(axis=0, keepdims=False)
                elif channel == "left":
                    dst_path = os.path.join(
                        cfg.data.cleaned_data_root,
                        cfg.dataset.name,
                        AUDIO,
                        subset,
                        split,
                        f"{track_id_str}_{short_class}{chunk_idx:03d}_left.wav",
                    )
                    chunk = audio_chunk[0, :]
                elif channel == "right":
                    dst_path = os.path.join(
                        cfg.data.cleaned_data_root,
                        cfg.dataset.name,
                        AUDIO,
                        subset,
                        split,
                        f"{track_id_str}_{short_class}{chunk_idx:03d}_right.wav",
                    )
                    chunk = audio_chunk[1, :]
                else:
                    raise ValueError(f"Unsupported channel: {channel}")

                metadata.append(
                    {
                        "file": src_path.replace(
                            cfg.data.raw_data_root, "$RAW_DATA_ROOT"
                        ),
                        "cleaned_path": dst_path.replace(
                            cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
                        ),
                        "start_time": start_time_,
                        "end_time": end_time_,
                        "channel": channel,
                        "license": license_,
                        "license_usage": license_usage,
                        "license_family": license_family,
                        "dsid_fma_track_id": track_id,
                        "subset": subset,
                        "smad_class": class_,
                    }
                )

                audios.append(chunk)
                dst_paths.append(dst_path)

            chunk_idx += 1

    if len(audios) == 0:
        return []

    with Pool(min(len(audios), os.cpu_count())) as p:
        p.starmap(soundfile_export, zip(audios, dst_paths, [cfg] * len(audios)))

    return metadata


def organize_file_chunked(
    track_id: int,
    license_: str,
    license_usage: str,
    license_family: str,
    split: str,
    raw_cfg: DictConfig,
    cfg: DictConfig,
):
    """Organizes an FMA track by splitting it into fixed-size chunks."""
    track_id_str = f"{track_id:06d}"
    folder_id = track_id_str[:3]
    src_path = os.path.join(
        cfg.data.raw_data_root, raw_cfg.path, folder_id, f"{track_id_str}.mp3"
    )

    chunk_size_seconds = cfg.fma.chunk_size_seconds
    subset = cfg.dataset.subset.format(license=license_usage, inst="unk")

    out = probe_and_check(src_path, subset, cfg)

    if out is not None:
        return out

    dst_dir = os.path.join(
        cfg.data.cleaned_data_root, cfg.dataset.name, AUDIO, subset, split
    )
    os.makedirs(dst_dir, exist_ok=True)

    audio = ffmpeg_reformat_to_buffer(
        file=src_path,
        sampling_rate=cfg.audio.sampling_rate,
        num_channels=2,
        channel_first=True,
    )

    n_samples = audio.shape[-1]
    duration = n_samples / cfg.audio.sampling_rate
    n_samples_per_chunk = int(chunk_size_seconds * cfg.audio.sampling_rate)

    n_full_chunks = int(np.floor(duration / chunk_size_seconds))

    if n_full_chunks == 0:
        assert duration < chunk_size_seconds
        return []

    assert n_full_chunks * n_samples_per_chunk <= n_samples

    downmix_mode = cfg.audio.get("downmix_mode", "mid")

    if downmix_mode == "auto":
        channels = ["mid"]
    elif downmix_mode == "lr":
        channels = ["left", "right"]
    else:
        raise ValueError(f"Unsupported downmix mode: {downmix_mode}")

    random_state = np.random.default_rng(seed=track_id)

    metadata = []
    audios = []
    dst_paths = []

    for channel in channels:
        remaining_samples = n_samples - n_full_chunks * n_samples_per_chunk
        prev_end_time = 0

        for i in range(n_full_chunks):
            assert remaining_samples >= 0

            start_time_offset = (
                random_state.integers(0, remaining_samples)
                if remaining_samples > 0
                else 0
            )
            start_time = prev_end_time + start_time_offset
            end_time = start_time + n_samples_per_chunk
            prev_end_time = end_time

            remaining_samples -= start_time_offset

            chunk = audio[:, start_time:end_time]

            if channel == "mid":
                dst_path = os.path.join(dst_dir, f"{track_id_str}_mid_{i:03d}.wav")
                chunk = chunk.mean(axis=0, keepdims=False)
            elif channel == "left":
                dst_path = os.path.join(dst_dir, f"{track_id_str}_left_{i:03d}.wav")
                chunk = chunk[0, :]
            elif channel == "right":
                dst_path = os.path.join(dst_dir, f"{track_id_str}_right_{i:03d}.wav")
                chunk = chunk[1, :]
            else:
                raise ValueError(f"Unsupported channel: {channel}")

            metadata.append(
                {
                    "file": src_path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
                    "cleaned_path": dst_path.replace(
                        cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
                    ),
                    "start_time": start_time / cfg.audio.sampling_rate,
                    "end_time": end_time / cfg.audio.sampling_rate,
                    "channel": channel,
                    "license": license_,
                    "license_usage": license_usage,
                    "license_family": license_family,
                    "dsid_fma_track_id": track_id,
                    "subset": subset,
                }
            )

            audios.append(chunk)
            dst_paths.append(dst_path)

    if len(audios) == 0:
        return []

    p = Pool(min(len(audios), os.cpu_count()))
    p.starmap(soundfile_export, zip(audios, dst_paths, [cfg] * len(audios)))
    p.close()
    p.join()

    return metadata


def probe_and_check(src_path, subset, cfg):
    """Probes a source file with ffmpeg to check for corruption."""
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"File not found: {src_path}")
        return

    try:
        ffmpeg.probe(src_path)
    except ffmpeg.Error as e:
        if "Format mp3 detected only with low score of" in str(e.stderr):
            print(f"{os.path.basename(src_path)}: Low score, skipping")
            return {
                "file": src_path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
                "cleaned_path": None,
                "error": "Format mp3 detected only with low score",
                "subset": subset,
            }
        else:
            print(f"Error during PROBE: for file: {src_path}")
            print(e.stderr)
            return {
                "file": src_path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
                "cleaned_path": None,
                "error": str(e),
                "subset": subset,
            }


def organize_file_unchunked(
    track_id: int,
    license_: str,
    license_usage: str,
    license_family: str,
    split: str,
    raw_cfg: DictConfig,
    cfg: DictConfig,
):
    """Organizes an FMA track by converting the full file without chunking."""
    track_id_str = f"{track_id:06d}"
    folder_id = track_id_str[:3]
    src_path = os.path.join(
        cfg.data.raw_data_root, raw_cfg.path, folder_id, f"{track_id_str}.mp3"
    )

    subset = cfg.dataset.subset.format(license=license_usage, inst="unk")

    out = probe_and_check(src_path, subset, cfg)

    if out is not None:
        return out

    dst_path = os.path.join(
        cfg.data.cleaned_data_root,
        cfg.dataset.name,
        AUDIO,
        subset,
        split,
        f"{track_id_str}.wav",
    )

    out = convert_audio_with_downmix(src_path, dst_path, subset, split, cfg)

    if out is not None:
        return out

    return {
        "file": src_path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
        "cleaned_path": dst_path.replace(
            cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
        ),
        "license": license_,
        "license_usage": license_usage,
        "license_family": license_family,
        "dsid_fma_track_id": track_id,
        "subset": subset,
    }


def organize_file(
    track_id: int,
    license_: str,
    license_usage: str,
    license_family: str,
    split: str,
    raw_cfg: DictConfig,
    cfg: DictConfig,
    smad=None,
) -> None:
    """
    Main dispatcher for organizing a single FMA file based on the configuration.

    Delegates to smad, chunked, or unchunked organization functions.
    """
    if cfg.fma.get("run_smad", False):
        return organize_file_smad(
            track_id=track_id,
            license_=license_,
            license_usage=license_usage,
            license_family=license_family,
            split=split,
            raw_cfg=raw_cfg,
            cfg=cfg,
            smad=smad,
        )

    if cfg.fma.get("chunk", False):
        return organize_file_chunked(
            track_id=track_id,
            license_=license_,
            license_usage=license_usage,
            license_family=license_family,
            split=split,
            raw_cfg=raw_cfg,
            cfg=cfg,
        )

    return organize_file_unchunked(
        track_id=track_id,
        license_=license_,
        license_usage=license_usage,
        license_family=license_family,
        split=split,
        raw_cfg=raw_cfg,
        cfg=cfg,
    )


def scan_valid_tracks(track_id, raw_cfg, cfg, overwrite=False):
    """Scans a track to see if it is valid and needs processing for SMAD."""
    track_id_str = f"{track_id:06d}"
    folder_id = track_id_str[:3]
    src_path = os.path.join(
        cfg.data.raw_data_root, raw_cfg.path, folder_id, f"{track_id_str}.mp3"
    )

    smad_path = src_path.replace(".mp3", "_smad.csv").replace(
        f"fma_{cfg.fma.subset}", f"fma_{cfg.fma.subset}_smad"
    )

    if not overwrite:
        if os.path.exists(smad_path):
            return None

    out = probe_and_check(src_path, None, cfg)

    if out is not None:
        return None

    return track_id


def run_smad_on_ids(track_ids, raw_cfg, cfg, position, overwrite=True):
    """Worker function to run SMAD on a list of track IDs."""
    smad = smad_predictor(cuda=False)

    for track_id in tqdm(track_ids, position=position):
        track_id_str = f"{track_id:06d}"
        folder_id = track_id_str[:3]
        src_path = os.path.join(
            cfg.data.raw_data_root, raw_cfg.path, folder_id, f"{track_id_str}.mp3"
        )

        try:
            get_smad_activations(src_path, smad, cfg, overwrite=overwrite)
        except Exception as e:
            print(f"Error for track: {track_id}")
            print(e)
            continue

    del smad


def run_smad(track_ids, raw_cfg, cfg, overwrite=True):
    """Runs SMAD in parallel on a list of track IDs."""
    valid_track_ids = process_map(
        scan_valid_tracks,
        track_ids,
        [raw_cfg] * len(track_ids),
        [cfg] * len(track_ids),
        [overwrite] * len(track_ids),
        chunksize=16,
    )

    valid_track_ids = [vti for vti in valid_track_ids if vti is not None]

    n_groups = 4
    n_tracks_per_group = int(math.ceil(len(valid_track_ids) / n_groups))

    valid_track_id_groups = [
        valid_track_ids[i : min(i + n_tracks_per_group, len(valid_track_ids))]
        for i in range(0, len(valid_track_ids), n_tracks_per_group)
    ]

    process_map(
        run_smad_on_ids,
        valid_track_id_groups,
        [raw_cfg] * len(valid_track_id_groups),
        [cfg] * len(valid_track_id_groups),
        range(len(valid_track_id_groups)),
        [overwrite] * len(valid_track_id_groups),
        chunksize=1,
        max_workers=n_groups,
    )


def organize_split(split: str, cfg: DictConfig, parallel=True) -> None:
    """Organizes all files for a single FMA dataset split."""
    fma_subset = cfg.fma.subset

    run_smad_first = False
    if cfg.fma.get("run_smad", False):
        if cfg.fma.get("smad_data_root", None) is None:
            run_smad_first = True
            print("Running smad first")

    manifest = []

    for raw_cfg in cfg.splits[split]:
        df = load_metadata(split=split, cfg=cfg, src_cfg=raw_cfg)

        return df

        if cfg.fma.get("license_filter", None) is not None:
            df = df[df.license_usage.isin(cfg.fma.license_filter)]

        track_ids = df.id.tolist()
        licenses = df.license.tolist()
        license_usage = df.license_usage.tolist()
        license_family = df.license_family.tolist()

        if run_smad_first:
            print(
                "Running SMAD only first. Set `smad_data_root` then come back to run the rest."
            )
            run_smad(track_ids, raw_cfg, cfg)
            continue

        if parallel:
            manifest += process_map(
                organize_file,
                track_ids,
                licenses,
                license_usage,
                license_family,
                [split] * len(track_ids),
                [raw_cfg] * len(track_ids),
                [cfg] * len(track_ids),
                chunksize=1,
                # max_workers=16
            )

        else:
            for track_id, license_, license_usage_, license_family_ in tqdm(
                zip(track_ids, licenses, license_usage, license_family)
            ):
                manifest += [
                    organize_file(
                        track_id=track_id,
                        license_=license_,
                        license_usage=license_usage_,
                        license_family=license_family_,
                        split=split,
                        raw_cfg=raw_cfg,
                        cfg=cfg,
                    )
                ]

    if cfg.fma.get("chunk", False) or cfg.fma.get("run_smad", False):
        manifest = list(chain(*manifest))

    manifest = pd.DataFrame(manifest)

    for subset, dfg in manifest.groupby("subset"):
        manifest_path = os.path.join(
            cfg.data.cleaned_data_root,
            cfg.dataset.name,
            MANIFEST,
            subset,
            f"{split}.csv",
        )

        failed_path = os.path.join(
            cfg.data.cleaned_data_root,
            cfg.dataset.name,
            MANIFEST,
            subset,
            f"{split}_failed.csv",
        )

        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        os.makedirs(os.path.dirname(failed_path), exist_ok=True)

        failed_dfg = dfg[dfg.cleaned_path.isna()]
        dfg = dfg.dropna(subset=["cleaned_path"])

        succeeded = len(dfg)
        failed = len(failed_dfg)

        print(f"Split: {split}, Subset: {subset}")
        print(f"Succeeded: {succeeded}, Failed: {failed}")
        print(f"Pct failed: {100 * failed / (succeeded + failed):.2f}")

        dfg.to_csv(manifest_path, index=False)
        failed_dfg.to_csv(failed_path, index=False)


def organize_fma(cfg: DictConfig) -> None:
    """Main entry point for organizing the FMA dataset."""
    preface(cfg)

    dfs = []

    for split in cfg.splits:
        print()
        print("".join(["="] * 80))
        print(f"Organizing split: {split}")
        print("".join(["="] * 80))
        print()

        df = organize_split(split=split, cfg=cfg)

        dfs.append(df)

    df = pd.concat(dfs)

    license_dtype = pd.CategoricalDtype(categories=["Y", "N", "?"], ordered=True)

    def check_deriv(x):
        if "cc" in x:
            if "sampling" in x:
                return "?"
            return "Y" if "nd" not in x else "N"
        else:
            if x in ["free-art-license", "open-audio-license", "public-domain"]:
                return "Y"
            if x in ["sound-recording-common-law", "orphan-work", "missing"]:
                return "?"
            return "N"

    def check_com(x):
        if "cc" in x:
            return "Y" if "nc" not in x else "N"
        else:
            if x in ["free-art-license", "open-audio-license", "public-domain"]:
                return "Y"
            if x in ["sound-recording-common-law", "orphan-work", "missing"]:
                return "?"
            return "N"

    # print(split)
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        300,
        "float_format",
        "{:2.3g}".format,
    ):
        df["deriv_"] = df["license_family"].apply(check_deriv).astype(license_dtype)
        df["com_"] = df["license_family"].apply(check_com).astype(license_dtype)

        summ = (
            df.value_counts(["com_", "deriv_", "license_family"])
            .reset_index()
            .sort_values(
                ["com_", "deriv_", "license_family", "count"],
                ascending=[True, True, True, False],
            )
            .set_index(["com_", "deriv_", "license_family"])
        )
        summ["pc"] = summ["count"] / len(df) * 100
        summ["pc"] = summ["pc"].apply(lambda x: f"{x:.1f}" if x > 0.1 else f"{x:.1g}")

        print(summ.to_latex(sparsify=True, multirow=False))
