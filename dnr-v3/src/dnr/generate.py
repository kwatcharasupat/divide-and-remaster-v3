import multiprocessing as mp
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import poisson
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ..const import (
    AUDIO,
    DEFAULT,
    MANIFEST,
    MIXTURE,
    NPY64,
)
from .audio_utils.loudness import adjust_audio_lufs, get_lufs, normalize_audio
from .audio_utils.mastering import apply_mastering
from .io import output
from .manifest import CloneSpec, StemSpec, get_manifest
from .processing import (
    _create_audio_event,
    get_time_range,
    load_and_select_segment,
    process_stems,
)
from .random import (
    _get_clip_lufs,
    _get_mix_lufs,
    _get_num_segments,
    _get_stem_lufs,
    _get_submix_lufs,
)


def get_file_and_check(
    curr_file_index: List[int],
    split: str,
    submix: str,
    subset: str,
    manifest: pd.DataFrame,
    cfg: DictConfig,
    random_state: np.random.Generator,
    min_start_seconds: float,
    submix_duration_seconds: float,
    submix_lufs: float = None,
) -> Tuple[int, int, int]:
    if curr_file_index is not None:
        if len(curr_file_index) == 0:
            if cfg.splits[split].exhaust[submix][subset].loop > 1:
                print("Exhausted subset once. Looping again.")
                curr_file_index = list(range(len(manifest)))
                random_state.shuffle(curr_file_index)

                cfg.splits[split].exhaust[submix][subset].loop -= 1
            else:
                return (None, None, None, None, -1, None, None, None, None, None)

        file_index = curr_file_index.pop(0)
    else:
        p = manifest["sample_prob"]
        file_index = random_state.choice(manifest.index, p=p)

    file = manifest.iloc[file_index]["cleaned_path"]
    try:
        file = os.path.expandvars(file)
    except:
        print(manifest.iloc[file_index])
        raise ValueError("File path not found")

    if "audiocite" in file:
        file = file.replace("slr139-audiocite/", "slr139-aucite/")
    npy_file = file.replace(AUDIO, NPY64).replace(".wav", ".npy")

    # print("File index: ", file_index)

    audio_length_samples = np.load(npy_file, mmap_mode="r").shape[-1]
    event_length_seconds = audio_length_samples / cfg.audio.sampling_rate

    assert min_start_seconds >= 0

    start_sample, length_sample = get_time_range(
        event_length_seconds=event_length_seconds,
        min_start_seconds=min_start_seconds,
        submix_duration_seconds=submix_duration_seconds,
        submix=submix,
        random_state=random_state,
        cfg=cfg,
    )

    if start_sample is None:
        if curr_file_index is not None:
            curr_file_index.append(file_index)
            random_state.shuffle(curr_file_index)
        return (
            None,
            file,
            start_sample,
            length_sample,
            curr_file_index,
            None,
            audio_length_samples,
            event_length_seconds,
            None,
            None,
        )

    audio_segment, segment_start_sample = load_and_select_segment(
        file_path=npy_file,
        length_sample=length_sample,
        random_segment_start=cfg.events[submix].random_segment_start,
        random_state=random_state,
    )

    clip_lufs = _get_clip_lufs(
        submix_lufs=submix_lufs, submix=submix, cfg=cfg, random_state=random_state
    )

    audio_segment, clip_lufs, _ = normalize_audio(audio_segment, clip_lufs, cfg=cfg)

    return (
        file_index,
        file,
        start_sample,
        length_sample,
        curr_file_index,
        audio_segment,
        audio_length_samples,
        event_length_seconds,
        segment_start_sample,
        clip_lufs,
    )


def truncated_poisson(
    lambda_: float, min_val: int, max_val: int, random_state: np.random.Generator
) -> int:
    low_cut = poisson.cdf(min_val, lambda_)
    high_cut = poisson.cdf(max_val, lambda_)

    p = random_state.uniform(low_cut, high_cut)

    return int(poisson.ppf(p, lambda_))


def get_stems_and_check(
    split: str,
    submix: str,
    subset: str,
    manifest: StemSpec,
    cfg: DictConfig,
    random_state: np.random.Generator,
    min_start_seconds: float,
    submix_duration_seconds: float,
    submix_lufs: float = None,
) -> Tuple[int, int, int]:
    track_index = random_state.integers(0, manifest.length, endpoint=False)
    track_dict = manifest.subset_manifests[track_index]

    substem_source_dict = {}

    for substem, components in track_dict.items():
        total_components = len(components)

        if cfg.events[submix].n_components.distr == "uniform":
            n_components = random_state.integers(1, total_components, endpoint=True)
        elif cfg.events[submix].n_components.distr == "truncated-poisson":
            n_components = truncated_poisson(
                lambda_=cfg.events[submix].n_components.loc_multiplier
                * total_components,
                min_val=1,
                max_val=total_components,
                random_state=random_state,
            )
        else:
            raise ValueError(
                "Invalid distribution for n_components: ",
                cfg.events[submix].n_components.distr,
            )

        active_components = random_state.choice(
            list(components.keys()), n_components, replace=False
        )

        substem_sources = {
            component: random_state.choice(components[component])
            for component in active_components
        }

        substem_source_dict[substem] = substem_sources

        dummy_file = list(substem_sources.values())[0]

    try:
        file = os.path.expandvars(dummy_file)
    except:
        raise ValueError("File path not found: ", dummy_file)

    npy_file = file.replace(AUDIO, NPY64).replace(".wav", ".npy")

    audio = np.load(npy_file, mmap_mode="r")

    audio_length_samples = audio.shape[-1]
    event_length_seconds = audio_length_samples / cfg.audio.sampling_rate

    assert min_start_seconds >= 0

    start_sample, length_sample = get_time_range(
        event_length_seconds=event_length_seconds,
        min_start_seconds=min_start_seconds,
        submix_duration_seconds=submix_duration_seconds,
        submix=submix,
        random_state=random_state,
        cfg=cfg,
    )

    if start_sample is None:
        return (
            None,
            file,
            start_sample,
            length_sample,
            None,
            None,
            audio_length_samples,
            event_length_seconds,
            None,
            None,
            None,
        )

    if cfg.events[submix].random_segment_start:
        segment_start_sample = random_state.integers(
            0, audio_length_samples - length_sample
        )
    else:
        segment_start_sample = 0

    audio_segment_dict = {}
    clip_lufs_dict = {}
    stem_lufs_dict = defaultdict(dict)
    clip_lufs = _get_clip_lufs(
        submix_lufs=submix_lufs,
        submix=submix,
        cfg=cfg,
        random_state=random_state,
    )

    for substem in manifest.substems:
        audio_segment = np.zeros((cfg.audio.channels, length_sample))

        if substem not in substem_source_dict:
            audio_segment_dict[substem] = audio_segment
            stem_lufs_dict[substem]["__nothing__"] = -np.inf
            continue

        for component, source in substem_source_dict[substem].items():
            npy_file = source.replace(AUDIO, NPY64).replace(".wav", ".npy")
            npy_file = os.path.expandvars(npy_file)
            audio = np.load(npy_file, mmap_mode="r")

            stem_segment = audio[
                :, segment_start_sample : segment_start_sample + length_sample
            ]

            if segment_start_sample + length_sample > audio.shape[-1]:
                # this is needed in rare cases for MoisesDB due to audio length inconsistencies
                stem_segment = np.pad(
                    stem_segment,
                    (
                        (0, 0),
                        (0, segment_start_sample + length_sample - audio.shape[-1]),
                    ),
                    mode="constant",
                )

            stem_lufs = _get_stem_lufs(
                clip_lufs=clip_lufs, submix=submix, cfg=cfg, random_state=random_state
            )

            stem_segment, stem_lufs, _ = normalize_audio(
                stem_segment, stem_lufs, cfg=cfg
            )

            audio_segment += stem_segment

            stem_lufs_dict[substem][component] = stem_lufs

        audio_segment_dict[substem] = audio_segment

    audio_segment_dict, clip_lufs_dict, effective_clip_gain = process_stems(
        audio_segment_dict, submix_lufs, cfg
    )

    for substem in manifest.substems:
        for component in stem_lufs_dict[substem]:
            stem_lufs_dict[substem][component] += effective_clip_gain

    return (
        track_index,
        substem_source_dict,
        start_sample,
        length_sample,
        None,
        audio_segment_dict,
        audio_length_samples,
        event_length_seconds,
        segment_start_sample,
        clip_lufs_dict,
        stem_lufs_dict,
    )


def _generate_stem_submix_subset(
    submix_lufs: float,
    num_seg: int,
    subset: str,
    submix: str,
    idx: int,
    split: str,
    manifest: pd.DataFrame,
    cfg: DictConfig,
    random_state: np.random.Generator,
    max_trials_lufs: int = 10,
):
    submix_duration_seconds = cfg.audio.duration
    submix_duration_samples = int(cfg.audio.sampling_rate * submix_duration_seconds)

    submix_audio_dict = {
        substem: np.zeros((cfg.audio.channels, submix_duration_samples))
        for substem in manifest.substems
    }
    submix_annots_dict = {substem: [] for substem in manifest.substems}

    min_start_sample = 0

    for j in range(num_seg):
        min_start_seconds = min_start_sample / cfg.audio.sampling_rate

        results, success = _create_audio_event(
            get_stems_and_check,
            max_trials=max_trials_lufs,
            split=split,
            submix=submix,
            subset=subset,
            manifest=manifest,
            cfg=cfg,
            random_state=random_state,
            min_start_seconds=min_start_seconds,
            submix_duration_seconds=cfg.audio.duration,
            submix_lufs=submix_lufs,
        )

        if not success:
            continue

        (
            file_index,
            file_dict,
            start_sample,
            length_sample,
            curr_file_index,
            audio_segment_dict,
            audio_length_samples,
            event_length_seconds,
            segment_start_sample,
            clip_lufs_dict,
            stem_lufs_dict,
        ) = results

        assert length_sample > 0
        assert np.isfinite(list(clip_lufs_dict.values())).all()

        for substem in stem_lufs_dict:
            audio_segment = audio_segment_dict[substem]
            # clip_lufs = clip_lufs_dict[substem]
            submix_audio_dict[substem][
                :, start_sample : start_sample + length_sample
            ] += audio_segment

            for component in stem_lufs_dict[substem]:
                submix_annots_dict[substem].append(
                    {
                        "file": file_dict[substem][component],
                        "start_sample": start_sample,
                        "length_sample": length_sample,
                        "segment_start_sample": segment_start_sample,
                        "start_seconds": start_sample / cfg.audio.sampling_rate,
                        "length_seconds": length_sample / cfg.audio.sampling_rate,
                        "end_seconds": (start_sample + length_sample)
                        / cfg.audio.sampling_rate,
                        "lufs": stem_lufs_dict[substem][component],
                    }
                )

        if cfg.events[submix].allow_overlap_prop > 0:
            skip_samples = length_sample * (1.0 - cfg.events[submix].allow_overlap_prop)
            min_start_sample = random_state.integers(
                start_sample + skip_samples,
                min(submix_duration_samples, start_sample + length_sample),
            )
        else:
            min_start_sample = start_sample + length_sample

    for substem in manifest.substems:
        submix_audio_dict[substem], submix_lufs_normalized, submix_lufs_original = (
            normalize_audio(submix_audio_dict[substem], submix_lufs, cfg=cfg)
        )

        effective_gain = submix_lufs_normalized - submix_lufs_original

        submix_annots_dict[substem] = pd.DataFrame(submix_annots_dict[substem])
        submix_annots_dict[substem]["lufs"] += effective_gain
        submix_annots_dict[substem]["submix_lufs"] = submix_lufs_normalized
        submix_annots_dict[substem]["submix_lufs_target"] = submix_lufs

    submix_audio_dict = {
        f"{submix}_{substem}": submix_audio_dict[substem]
        for substem in manifest.substems
    }

    submix_annots_dict = {
        f"{submix}_{substem}": submix_annots_dict[substem]
        for substem in manifest.substems
    }

    return submix_audio_dict, submix_annots_dict, None


def _generate_submix_subset(
    submix_lufs: float,
    num_seg: int,
    subset: str,
    submix: str,
    idx: int,
    split: str,
    manifest: pd.DataFrame,
    cfg: DictConfig,
    random_state: np.random.Generator,
    curr_file_index: int = None,
    max_trials_lufs: int = 10,
) -> None:
    submix_duration_seconds = cfg.audio.duration
    submix_duration_samples = int(cfg.audio.sampling_rate * submix_duration_seconds)

    submix_audio = np.zeros((cfg.audio.channels, submix_duration_samples))
    submix_annots = []

    min_start_sample = 0

    for j in range(num_seg):
        min_start_seconds = min_start_sample / cfg.audio.sampling_rate

        results, success = _create_audio_event(
            get_file_and_check,
            max_trials=max_trials_lufs,
            curr_file_index=curr_file_index,
            split=split,
            submix=submix,
            subset=subset,
            manifest=manifest,
            cfg=cfg,
            random_state=random_state,
            min_start_seconds=min_start_seconds,
            submix_duration_seconds=cfg.audio.duration,
            submix_lufs=submix_lufs,
        )

        if not success:
            continue

        (
            file_index,
            file,
            start_sample,
            length_sample,
            curr_file_index,
            audio_segment,
            audio_length_samples,
            event_length_seconds,
            segment_start_sample,
            clip_lufs,
        ) = results

        assert length_sample > 0
        assert np.isfinite(clip_lufs)

        submix_audio[:, start_sample : start_sample + length_sample] += audio_segment

        submix_annots.append(
            {
                "file": file,
                "start_sample": start_sample,
                "length_sample": length_sample,
                "segment_start_sample": segment_start_sample,
                "start_seconds": start_sample / cfg.audio.sampling_rate,
                "length_seconds": length_sample / cfg.audio.sampling_rate,
                "end_seconds": (start_sample + length_sample) / cfg.audio.sampling_rate,
                "lufs": clip_lufs,
            }
        )

        if cfg.events[submix].allow_overlap_prop > 0:
            skip_samples = length_sample * (1.0 - cfg.events[submix].allow_overlap_prop)
            min_start_sample = random_state.integers(
                start_sample + skip_samples,
                min(submix_duration_samples, start_sample + length_sample),
            )
        else:
            min_start_sample = start_sample + length_sample

    submix_audio, submix_lufs_normalized, submix_lufs_original = normalize_audio(
        submix_audio, submix_lufs, cfg=cfg
    )

    effective_gain = submix_lufs_normalized - submix_lufs_original

    submix_annots = pd.DataFrame(submix_annots)
    submix_annots["lufs"] += effective_gain
    submix_annots["submix_lufs"] = submix_lufs_normalized
    submix_annots["submix_lufs_target"] = submix_lufs

    return submix_audio, submix_annots, curr_file_index


def _clone_submix_subset(
    idx: int,
    split: str,
    manifest: pd.DataFrame,
    cfg: DictConfig,
):
    clone_src = manifest.src
    clone_subset = manifest.subset
    clone_submix = manifest.submix

    submix_audio_path = os.path.join(
        os.path.expandvars(clone_src),
        AUDIO,
        clone_subset,
        split,
        f"{idx:06d}",
        f"{clone_submix}.npy",
    )

    if not os.path.exists(submix_audio_path):
        submix_audio_path = os.path.join(
            os.path.expandvars(clone_src),
            NPY64,
            clone_subset,
            split,
            f"{idx:06d}",
            f"{clone_submix}.npy",
        )

    assert os.path.exists(
        submix_audio_path
    ), f"Submix audio path does not exist: {submix_audio_path}"

    submix_annots_path = os.path.join(
        os.path.expandvars(clone_src),
        MANIFEST,
        clone_subset,
        split,
        f"{idx:06d}",
        f"{clone_submix}.csv",
    )

    submix_audio = np.load(submix_audio_path)
    submix_annots = pd.read_csv(submix_annots_path)

    return submix_audio, submix_annots, None


def generate_submix_subset(
    submix_lufs: float,
    num_seg: int,
    subset: str,
    submix: str,
    idx: int,
    split: str,
    manifest: pd.DataFrame,
    cfg: DictConfig,
    random_state: np.random.Generator,
    curr_file_index: int = None,
    max_trials_lufs: int = 10,
) -> None:
    if isinstance(manifest, CloneSpec):
        assert curr_file_index is None
        return _clone_submix_subset(
            idx=idx,
            split=split,
            manifest=manifest,
            cfg=cfg,
        )
    elif isinstance(manifest, StemSpec):
        assert curr_file_index is None
        return _generate_stem_submix_subset(
            submix_lufs,
            num_seg,
            subset=subset,
            submix=submix,
            idx=idx,
            split=split,
            manifest=manifest,
            cfg=cfg,
            random_state=random_state,
            max_trials_lufs=max_trials_lufs,
        )
    else:
        return _generate_submix_subset(
            submix_lufs,
            num_seg,
            subset=subset,
            submix=submix,
            idx=idx,
            split=split,
            manifest=manifest,
            cfg=cfg,
            random_state=random_state,
            curr_file_index=curr_file_index,
            max_trials_lufs=max_trials_lufs,
        )


def generate_submix(
    submix: str,
    idx: int,
    split: str,
    manifest_dict: Dict[str, pd.DataFrame],
    cfg: DictConfig,
    random_state: np.random.Generator,
    curr_file_index_dict: Dict[str, int] = None,
) -> None:
    """Generates a single submix (e.g., all speech events) for one mixture.

    Args:
        submix: The name of the submix to generate (e.g., "speech").
        idx: The index of the mixture file being generated.
        split: The name of the current dataset split.
        manifest_dict: A dictionary containing the data manifests for this submix.
        cfg: The Hydra configuration object.
        random_state: The random number generator for this submix.
        curr_file_index_dict: State for tracking file usage in exhaust mode.
    """
    submix_audio = {}
    submix_annots = {}

    submix_ = submix

    if cfg.loudness[submix_].submix.distr == "precomputed":
        lufs_file = cfg.loudness[submix_].submix.file.format(split=split)
        lufs_df = pd.read_csv(os.path.expandvars(lufs_file)).set_index("idx", drop=True)
        submix_lufs = lufs_df.iloc[idx][cfg.loudness[submix].submix.column].item()
    else:
        submix_lufs = _get_submix_lufs(
            submix=submix_, cfg=cfg, random_state=random_state
        )

    if cfg.events[submix_].num_segments.distr == "precomputed":
        num_seg_file = cfg.events[submix_].num_segments.file.format(split=split)

        if num_seg_file == lufs_file:
            num_seg_df = lufs_df
        else:
            num_seg_df = pd.read_csv(os.path.expandvars(num_seg_file)).set_index(
                "idx", drop=True
            )

        num_seg = int(
            num_seg_df.iloc[idx][cfg.events[submix_].num_segments.column].item()
        )

    else:
        num_seg = _get_num_segments(submix=submix_, cfg=cfg, random_state=random_state)

    for subset in manifest_dict:
        submix_subset_audio, submix_subset_annots, curr_file_index = (
            generate_submix_subset(
                submix_lufs=submix_lufs,
                num_seg=num_seg,
                subset=subset,
                submix=submix,
                idx=idx,
                split=split,
                manifest=manifest_dict[subset],
                cfg=cfg,
                random_state=random_state,
                curr_file_index=(
                    curr_file_index_dict.get(subset, None)
                    if curr_file_index_dict is not None
                    else None
                ),
            )
        )

        if curr_file_index == -1:
            return None, None, -1

        submix_audio[subset] = submix_subset_audio
        submix_annots[subset] = submix_subset_annots

        if curr_file_index is not None:
            curr_file_index_dict[subset] = curr_file_index

    return submix_audio, submix_annots, curr_file_index_dict


def generate_submixes(
    idx: int,
    split: str,
    subsets: List[str],
    manifest_dict: Dict[str, pd.DataFrame],
    cfg: DictConfig,
    random_state: np.random.Generator,
    curr_file_index_dict_dict: Dict[str, Dict[str, int]] = None,
    parallel: bool = True,
):
    if parallel:
        audio_dict, annots_dict, curr_file_index_dict_dict = (
            _generate_submixes_parallel(
                idx=idx,
                split=split,
                subsets=subsets,
                manifest_dict=manifest_dict,
                cfg=cfg,
                random_state=random_state,
                curr_file_index_dict_dict=curr_file_index_dict_dict,
            )
        )
    else:
        audio_dict, annots_dict, curr_file_index_dict_dict = (
            _generate_submixes_sequential(
                idx=idx,
                split=split,
                subsets=subsets,
                manifest_dict=manifest_dict,
                cfg=cfg,
                random_state=random_state,
                curr_file_index_dict_dict=curr_file_index_dict_dict,
            )
        )

    if cfg.get("intermediates_mode", False):
        for submix in audio_dict:
            for subset in audio_dict[submix]:
                audio_path = os.path.join(
                    cfg.output_dir, AUDIO, subset, split, f"{idx:06d}", f"{submix}.npy"
                )
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)

                np.save(audio_path, audio_dict[submix][subset])

                annots_path = os.path.join(
                    cfg.output_dir,
                    MANIFEST,
                    subset,
                    split,
                    f"{idx:06d}",
                    f"{submix}.csv",
                )
                os.makedirs(os.path.dirname(annots_path), exist_ok=True)
                pd.DataFrame(annots_dict[submix][subset]).to_csv(
                    annots_path, index=False
                )

        return None, None, curr_file_index_dict_dict

    return audio_dict, annots_dict, curr_file_index_dict_dict


def _generate_submixes_parallel(
    idx: int,
    split: str,
    subsets: List[str],
    manifest_dict: Dict[str, pd.DataFrame],
    cfg: DictConfig,
    random_state: np.random.Generator,
    curr_file_index_dict_dict: Dict[str, Dict[str, int]] = None,
):
    audio_dict = {}
    annots_dict = {}

    submixes = [submix for submix in cfg.data]
    idxs = [idx] * len(cfg.data)
    splits = [split] * len(cfg.data)
    manifest_dicts = [manifest_dict[submix] for submix in submixes]
    cfgs = [cfg] * len(cfg.data)
    random_states = random_state.spawn(len(cfg.data))
    curr_file_index_dicts = [
        curr_file_index_dict_dict.get(submix, None)
        if curr_file_index_dict_dict is not None
        else None
        for submix in cfg.data
    ]

    with mp.Pool() as pool:
        out = pool.starmap(
            generate_submix,
            zip(
                submixes,
                idxs,
                splits,
                manifest_dicts,
                cfgs,
                random_states,
                curr_file_index_dicts,
            ),
            chunksize=1,
        )

    for i, submix in enumerate(cfg.data):
        audio_dict[submix], annots_dict[submix], curr_file_index_dict = out[i]

        if curr_file_index_dict == -1:
            return None, None, -1

        if curr_file_index_dict is not None:
            curr_file_index_dict_dict[submix] = curr_file_index_dict

    return audio_dict, annots_dict, curr_file_index_dict_dict


def _generate_submixes_sequential(
    idx: int,
    split: str,
    subsets: List[str],
    manifest_dict: Dict[str, pd.DataFrame],
    cfg: DictConfig,
    random_state: np.random.Generator,
    curr_file_index_dict_dict: Dict[str, Dict[str, int]] = None,
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, pd.DataFrame]],
    Dict[str, Dict[str, int]],
]:
    audio_dict = {}
    annots_dict = {}

    for submix in cfg.data:
        submix_audio_dict, submix_annots_dict, curr_file_index_dict = generate_submix(
            submix=submix,
            idx=idx,
            split=split,
            manifest_dict=manifest_dict[submix],
            cfg=cfg,
            random_state=random_state,
            curr_file_index_dict=(
                curr_file_index_dict_dict.get(submix, None)
                if curr_file_index_dict_dict is not None
                else None
            ),
        )

        if curr_file_index_dict == -1:
            return None, None, -1

        if curr_file_index_dict is not None:
            curr_file_index_dict_dict[submix] = curr_file_index_dict

        audio_dict[submix] = submix_audio_dict
        annots_dict[submix] = submix_annots_dict

    return audio_dict, annots_dict, curr_file_index_dict_dict


def master_subset_and_export(
    idx: int,
    audio_dict: Dict[str, Dict[str, np.ndarray]],
    annots_dict: Dict[str, Dict[str, pd.DataFrame]],
    split: str,
    subset: str,
    manifest_dict: Dict[str, pd.DataFrame],
    cfg: DictConfig,
    random_state: np.random.Generator,
) -> None:
    """Applies final mastering to a generated mixture and saves all outputs.

    This function takes all the generated submixes for a single output file,
    creates the final mixture, applies mastering effects (e.g., limiting),
    and saves all audio files and their corresponding annotation manifests.

    Args:
        idx: The index of the mixture file being generated.
        audio_dict: A dictionary containing all the generated submix audio data.
        annots_dict: A dictionary containing all the generated annotation data.
        split: The name of the current dataset split.
        subset: The specific data subset this file belongs to.
        manifest_dict: A dictionary containing the original data manifests.
        cfg: The Hydra configuration object.
        random_state: The random number generator for the mastering process.
    """
    final_audio_dict = {}
    final_annots_dict = {}
    final_audio_param_dict = {}

    mixture = []
    for submix in audio_dict:
        if subset not in audio_dict[submix]:
            submix_audio = audio_dict[submix][DEFAULT]
            submix_annot = annots_dict[submix][DEFAULT]
        else:
            submix_audio = audio_dict[submix][subset]
            submix_annot = annots_dict[submix][subset]
        # submix_annot = annots_dict[submix].get(subset, annots_dict[submix][DEFAULT])
        mixture.append(submix_audio)

        final_audio_dict[submix] = submix_audio
        final_annots_dict[submix] = submix_annot

    mixture = sum(mixture)
    final_audio_dict[MIXTURE] = mixture

    lufs_dict = {
        submix: submix_annot["submix_lufs_target"].values[0]
        for submix, submix_annot in final_annots_dict.items()
    }

    final_audio_dict, final_audio_param_dict = apply_mastering(
        final_audio_dict,
        lufs_dict,
        cfg=cfg,
        random_state=random_state,
        idx=idx,
        split=split,
    )

    for composite in cfg.composite:
        comp_audio = []
        for submix in cfg.composite[composite]:
            comp_audio.append(final_audio_dict[submix])

        comp_audio = sum(comp_audio)

        final_audio_dict[composite] = comp_audio

    output(
        final_audio_dict=final_audio_dict,
        final_annots_dict=final_annots_dict,
        final_audio_param_dict=final_audio_param_dict,
        idx=idx,
        split=split,
        submix=submix,
        subset=subset,
        cfg=cfg,
    )

    return final_audio_dict, final_annots_dict, final_audio_param_dict


def generate_mixture(
    idx: int,
    split: str,
    subsets: List[str],
    manifest_dict: Dict[str, Dict[str, pd.DataFrame]],
    cfg: DictConfig,
    random_state: np.random.Generator,
    output_now: bool = True,
    parallel_submixes: bool = False,
    curr_file_index_dict_dict: Dict[str, Dict[str, int]] = None,
) -> None:
    """Generates a single audio mixture containing multiple submixes.

    This function orchestrates the creation of all submixes (e.g., speech, music)
    for a single output file and then masters them together into the final mixture.

    Args:
        idx: The index of the mixture file to generate.
        split: The name of the current dataset split.
        subsets: A list of all possible data subsets.
        manifest_dict: A dictionary containing the data manifests.
        cfg: The Hydra configuration object.
        random_state: The random number generator for this mixture.
        output_now: If True, saves the final audio immediately. If False, returns it.
        parallel_submixes: If True, generates submixes in parallel.
        curr_file_index_dict_dict: State for tracking file usage in exhaust mode.
    """
    audio_dict, annots_dict, curr_file_index_dict_dict = generate_submixes(
        idx=idx,
        split=split,
        subsets=subsets,
        manifest_dict=manifest_dict,
        cfg=cfg,
        random_state=random_state,
        curr_file_index_dict_dict=curr_file_index_dict_dict,
        parallel=parallel_submixes,
    )

    flattend_audio_dict = defaultdict(dict)
    flattend_annots_dict = defaultdict(dict)

    for submix in audio_dict:
        for subset in audio_dict[submix]:
            if isinstance(audio_dict[submix][subset], dict):
                for key in audio_dict[submix][subset]:
                    flattend_audio_dict[key][subset] = audio_dict[submix][subset][key]
                    flattend_annots_dict[key][subset] = annots_dict[submix][subset][key]
            else:
                flattend_audio_dict[submix][subset] = audio_dict[submix][subset]
                flattend_annots_dict[submix][subset] = annots_dict[submix][subset]

    audio_dict = flattend_audio_dict
    annots_dict = flattend_annots_dict

    if curr_file_index_dict_dict == -1:
        return -1, None, None

    if cfg.get("intermediates_mode", False):
        return curr_file_index_dict_dict, None, None

    if output_now:
        master_subsets_and_export(
            idx=idx,
            audio_dict=audio_dict,
            annots_dict=annots_dict,
            split=split,
            subsets=subsets,
            manifest_dict=manifest_dict,
            cfg=cfg,
            random_state=random_state,
        )

        return curr_file_index_dict_dict, None, None

    return (
        curr_file_index_dict_dict,
        audio_dict,
        annots_dict,
    )


def master_subsets_and_export(
    idx: int,
    audio_dict: Dict[str, Dict[str, np.ndarray]],
    annots_dict: Dict[str, Dict[str, pd.DataFrame]],
    split: str,
    subsets: List[str],
    manifest_dict: Dict[str, pd.DataFrame],
    cfg: DictConfig,
    random_state: np.random.Generator,
) -> None:
    for subset in subsets:
        master_subset_and_export(
            idx=idx,
            audio_dict=audio_dict,
            annots_dict=annots_dict,
            split=split,
            subset=subset,
            manifest_dict=manifest_dict,
            cfg=cfg,
            random_state=random_state,
        )


def precompute(
    split, split_cfg: DictConfig, cfg: DictConfig, random_state: np.random.Generator
) -> None:
    n_files = split_cfg.num_files

    df = []

    for i in tqdm(range(n_files)):
        speech_lufs = _get_submix_lufs(
            submix="speech", cfg=cfg, random_state=random_state
        )
        master_lufs = _get_mix_lufs(cfg=cfg, random_state=random_state)
        speech_segments = _get_num_segments(
            submix="speech", cfg=cfg, random_state=random_state
        )

        df += [
            {
                "idx": i,
                "speech_lufs": speech_lufs,
                "master_lufs": master_lufs,
                "speech_segments": speech_segments,
            }
        ]

    df = pd.DataFrame(df)

    os.makedirs(os.path.join(cfg.output_dir, split), exist_ok=True)

    df.to_csv(os.path.join(cfg.output_dir, split, "precompute.csv"), index=False)


def generate_split(
    split: str, cfg: DictConfig, random_state: np.random.Generator
) -> None:
    """Generates a complete dataset split (e.g., train, val, test).

    This is the main function for generating a split. It handles manifest loading,
    determines the number of files to generate, and orchestrates the mixture
    generation process, either sequentially or in parallel.

    Args:
        split: The name of the split to generate (e.g., "train").
        cfg: The Hydra configuration object for the entire run.
        random_state: The master random number generator for this split.
    """
    split_cfg = cfg.splits[split]

    precompute_mode = cfg.get("precompute_mode", False)
    if precompute_mode:
        print("Precompute mode")
        precompute(split, split_cfg, cfg, random_state)
        return

    print(f"Generating split: {split}")

    manifest_dict, subsets = get_manifest(split=split, cfg=cfg)

    print(f"Manifest indicates a total of {len(subsets)} subsets")
    print(f"Subsets: {subsets}")

    intermediates_mode = split_cfg.get("intermediates_mode", False)
    output_now = not intermediates_mode

    if split_cfg.num_files is not None:
        random_states = random_state.spawn(split_cfg.num_files)

        process_map(
            generate_mixture,
            range(split_cfg.num_files),
            [split] * split_cfg.num_files,
            [subsets] * split_cfg.num_files,
            [manifest_dict] * split_cfg.num_files,
            [cfg] * split_cfg.num_files,
            random_states,
            [output_now] * split_cfg.num_files,
            chunksize=1,
        )
    else:
        curr_file_index_dict_dict = {}
        max_file_index_dict_dict = {}

        total_files = 0

        exhausting_submix = None

        if "exhaust" in split_cfg:
            for submix in split_cfg.exhaust:
                curr_file_index_dict_dict[submix] = {}
                max_file_index_dict_dict[submix] = {}
                print(split_cfg.exhaust[submix])
                for subset in split_cfg.exhaust[submix]:
                    curr_file_index_dict_dict[submix][subset] = list(
                        range(len(manifest_dict[submix][subset]))
                    )

                    random_state.shuffle(curr_file_index_dict_dict[submix][subset])

                    max_file_index_dict_dict[submix][subset] = len(
                        manifest_dict[submix][subset]
                    )
                    total_files = len(manifest_dict[submix][subset])

                    exhausting_submix = submix

                    break  # allow only one subset for now
                break  # allow only one submix for now

        cfg_clone = cfg.copy()
        cfg_clone.data = {
            submix: cfg.data[submix]
            for submix in cfg.data
            if submix != exhausting_submix
        }

        cfg_exhaust = cfg.copy()
        cfg_exhaust.data = {exhausting_submix: cfg.data[exhausting_submix]}

        pbar = tqdm(total=total_files)
        i = 0

        audio_dicts = []
        annots_dicts = []

        while True:
            (
                curr_file_index_dict_dict,
                audio_dict,
                annots_dict,
            ) = generate_mixture(
                idx=i,
                split=split,
                subsets=subsets,
                manifest_dict=manifest_dict,
                cfg=cfg_exhaust,
                random_state=random_state,
                output_now=False,
                parallel_submixes=False,
                curr_file_index_dict_dict=curr_file_index_dict_dict,
            )

            i += 1
            pbar.update(1)

            if curr_file_index_dict_dict == -1:
                print("Exhausted all files")
                print("Total files: ", i)
                break

            audio_dicts.append(audio_dict)
            annots_dicts.append(annots_dict)

        pbar.close()

        i -= 1

        outs = process_map(
            generate_mixture,
            range(i),
            [split] * i,
            [subsets] * i,
            [manifest_dict] * i,
            [cfg_clone] * i,
            random_state.spawn(i),
            [False] * i,
            chunksize=1,
        )

        _, audio_dicts_, annots_dicts_ = zip(*outs)

        print(len(audio_dicts_), len(audio_dicts), i)

        for j in tqdm(range(i)):
            audio_dicts[j].update(audio_dicts_[j])
            annots_dicts[j].update(annots_dicts_[j])

        random_states = random_state.spawn(i)

        process_map(
            master_subsets_and_export,
            range(i),
            audio_dicts,
            annots_dicts,
            [split] * i,
            [subsets] * i,
            [manifest_dict] * i,
            [cfg] * i,
            random_states,
            chunksize=1,
        )
