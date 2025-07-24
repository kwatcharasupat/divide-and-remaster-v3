from typing import Dict, Tuple, Union

import numpy as np
from omegaconf import DictConfig
from scipy import stats

from .contrib.zt import ztpoisson


def _get_rvs_kwargs(
    distr: str, loc: float, scale: float
) -> Tuple[Union[stats.rv_continuous, stats.rv_discrete], Dict[str, float]]:
    """Gets the appropriate scipy.stats object and parameters for a given distribution."""
    if distr == "normal":
        return stats.norm, {
            "loc": loc,
            "scale": scale,
        }

    if distr == "uniform":
        # The uniform distribution is defined on the interval [loc, loc+scale]
        # We need to convert it to the interval [loc-scale, loc+scale]
        return stats.uniform, {
            "loc": loc - scale,
            "scale": 2 * scale,
        }

    if distr == "zero-truncated-poisson":
        return ztpoisson, {"mu": loc}

    raise ValueError(f"Unknown distribution: {distr}")


def _get_start_seconds(
    min_start_seconds: float,
    max_start_seconds: float,
    submix: str,
    cfg: DictConfig,
    random_state: np.random.Generator,
):
    """Generates a random start time for an event based on a skew-normal distribution."""
    # hard coded distribution for now

    start_seconds_distr = cfg.events[submix].start_seconds.distr
    assert start_seconds_distr == "skew-normal"

    loc_offset = cfg.events[submix].start_seconds.loc_offset
    scale = cfg.events[submix].start_seconds.scale
    skew = cfg.events[submix].start_seconds.skew

    rvs_obj = stats.skewnorm
    rvs_kwargs = dict(loc=min_start_seconds + loc_offset, scale=scale, a=skew)

    out = rvs_obj.rvs(**rvs_kwargs, size=1, random_state=random_state)[0]

    out = max(0, out)

    return out


def _get_length_seconds(
    max_length_seconds: float,
    min_length_seconds: float,
    submix: str,
    cfg: DictConfig,
    random_state: np.random.Generator,
):
    """Generates a random length for an event based on a truncated-normal distribution."""
    length_seconds_distr = cfg.events[submix].length_seconds.distr
    assert length_seconds_distr == "truncated-normal"

    if "loc_multiplier" not in cfg.events[submix].length_seconds:
        assert "loc" in cfg.events[submix].length_seconds
        loc = cfg.events[submix].length_seconds.loc
    else:
        loc_mult = cfg.events[submix].length_seconds.loc_multiplier
        loc = loc_mult * max_length_seconds

    if "scale_multiplier" not in cfg.events[submix].length_seconds:
        assert "scale" in cfg.events[submix].length_seconds
        scale = cfg.events[submix].length_seconds.scale
    else:
        scale_mult = cfg.events[submix].length_seconds.scale_multiplier
        scale = scale_mult * max_length_seconds

    low = max(cfg.events[submix].length_seconds.lower_bound, min_length_seconds)

    if "upper_bound" in cfg.events[submix].length_seconds:
        high = min(cfg.events[submix].length_seconds.upper_bound, max_length_seconds)
    else:
        high = max_length_seconds

    if high <= low:
        assert high > 0
        return high

    rvs_obj = stats.truncnorm
    rvs_kwargs = dict(
        loc=loc, scale=scale, a=(low - loc) / scale, b=(high - loc) / scale
    )

    out = rvs_obj.rvs(**rvs_kwargs, size=1, random_state=random_state)[0]

    assert out > 0

    return out


def _get_submix_lufs(
    submix: str, cfg: DictConfig, random_state: np.random.Generator
) -> float:
    """Generates a random target LUFS for a submix."""
    submix_lufs_loc = cfg.loudness[submix].submix.loc
    submix_lufs_scale = cfg.loudness[submix].submix.scale
    submix_lufs_distr = cfg.loudness[submix].submix.distr

    rvs_obj, submix_rvs_kwargs = _get_rvs_kwargs(
        distr=submix_lufs_distr, loc=submix_lufs_loc, scale=submix_lufs_scale
    )

    return rvs_obj.rvs(**submix_rvs_kwargs, size=1, random_state=random_state)[0]


def _get_clip_lufs(
    submix_lufs: float, submix: str, cfg: DictConfig, random_state: np.random.Generator
):
    """Generates a random target LUFS for an individual clip within a submix."""
    clip_lufs_distr = cfg.loudness[submix].clip.distr
    clip_lufs_scale = cfg.loudness[submix].clip.scale

    rvs_obj, clip_rvs_kwargs = _get_rvs_kwargs(
        distr=clip_lufs_distr, loc=submix_lufs, scale=clip_lufs_scale
    )

    return rvs_obj.rvs(**clip_rvs_kwargs, size=1, random_state=random_state)[0]


def _get_stem_lufs(
    clip_lufs: float, submix: str, cfg: DictConfig, random_state: np.random.Generator
):
    """Generates a random target LUFS for a single stem within a clip."""
    stem_lufs_distr = cfg.loudness[submix].stem.distr
    stem_lufs_scale = cfg.loudness[submix].stem.scale

    rvs_obj, stem_rvs_kwargs = _get_rvs_kwargs(
        distr=stem_lufs_distr, loc=clip_lufs, scale=stem_lufs_scale
    )

    return rvs_obj.rvs(**stem_rvs_kwargs, size=1, random_state=random_state)[0]


def _get_mix_lufs(cfg: DictConfig, random_state: np.random.Generator) -> float:
    """Generates a random target LUFS for the final mastered mixture."""
    mix_lufs_distr = cfg.loudness.master.distr
    mix_lufs_loc = cfg.loudness.master.loc
    mix_lufs_scale = cfg.loudness.master.scale

    rvs_obj, mix_rvs_kwargs = _get_rvs_kwargs(
        distr=mix_lufs_distr, loc=mix_lufs_loc, scale=mix_lufs_scale
    )

    return rvs_obj.rvs(**mix_rvs_kwargs, size=1, random_state=random_state)[0]


def _get_num_segments(
    submix: str, cfg: DictConfig, random_state: np.random.Generator
) -> int:
    """Generates a random number of segments (events) for a submix."""
    num_seg_loc = cfg.events[submix].num_segments.loc
    num_seg_scale = cfg.events[submix].num_segments.get("scale", None)
    num_seg_distr = cfg.events[submix].num_segments.distr

    rvs_obj, num_seg_rvs_kwargs = _get_rvs_kwargs(
        distr=num_seg_distr, loc=num_seg_loc, scale=num_seg_scale
    )

    return int(rvs_obj.rvs(**num_seg_rvs_kwargs, size=1, random_state=random_state)[0])
