import os
from pprint import pprint
from typing import List

import pandas as pd
from omegaconf import DictConfig

from ..const import DEFAULT, MANIFEST


def clean_gender(g):
    """Normalizes gender strings to a consistent format."""
    if g is None:
        return "unknown"

    g = str(g).lower().strip()

    if g in ["m", "male"]:
        return "male"

    if g in ["f", "female"]:
        return "female"

    if g in ["nan", "unknown"]:
        return "unknown"

    raise ValueError(f"Unknown: {g}")


class CloneSpec:
    """Specification for cloning a pre-existing submix."""

    def __init__(self, src: str, subset: str, submix: str):
        super().__init__()
        self.src = src
        self.subset = subset
        self.submix = submix


class StemSpec:
    """Specification for generating a submix from multiple source stems."""

    def __init__(self, subset_sources, split, cfg):
        super().__init__()

        subset_manifests = []
        substems = set()

        for subset_spec in subset_sources:
            subset_manifest = []
            src_name = subset_spec.path
            for stem_group, stems in subset_spec.stems.items():
                for stem in stems:
                    stem_src_name = f"{src_name}/{stem}"
                    stem_manifest = get_subset_manifest(
                        src_name=stem_src_name, split=split, cfg=cfg
                    )

                    if stem_manifest is None:
                        continue

                    stem_manifest["stem_group"] = stem_group
                    stem_manifest["stem"] = stem
                    stem_manifest["src"] = src_name
                    subset_manifest.append(stem_manifest)

            subset_manifest = pd.concat(subset_manifest).reset_index(drop=True)

            substems = substems.union(subset_manifest["stem_group"].unique())

            # print(subset_manifest["file"].unique())

            subset_manifest["file"] = subset_manifest["file"].apply(
                lambda x: x.split("/")[-1]
            )

            subset_manifest = (
                subset_manifest.groupby("file")[["stem_group", "stem", "cleaned_path"]]
                .apply(
                    lambda x: x.groupby("stem_group")[["stem", "cleaned_path"]]
                    .apply(
                        lambda y: y.groupby("stem")["cleaned_path"]
                        .apply(list)
                        .to_dict()
                    )
                    .to_dict()
                )
                .to_list()
            )

            subset_manifests += subset_manifest

        self.subset_manifests = subset_manifests
        self.length = len(subset_manifests)
        self.substems = substems

        print("StemSpec: ", self.length, "with", self.substems, "substems")
        print("Substems: ", self.substems)


def load_manifest(src_name: str, split: str, cfg: DictConfig) -> List[str]:
    """Loads and filters a manifest file for a specific split and subset."""
    data_root = cfg.data_root

    dataset_details = src_name.split("/")
    top_level = dataset_details[0]
    sub_level = "/".join(dataset_details[1:])

    manifest_path = os.path.join(
        data_root, top_level, MANIFEST, sub_level, f"{split}.csv"
    )

    if not os.path.exists(manifest_path):
        print(f"Manifest path does not exist: {manifest_path}")
        return None

    df = pd.read_csv(manifest_path)

    if "subset" not in df:
        if "full/48k" in src_name:
            df["subset"] = "full/48k"
        else:
            raise ValueError("No subset in manifest")

    df = df[df["subset"] == sub_level]

    return df


def get_subset_manifest(src_name: str, split: str, cfg: DictConfig) -> List[str]:
    """Helper function to load a manifest for a given source name."""
    manifest = load_manifest(src_name=src_name, split=split, cfg=cfg)

    return manifest


def lang_to_family(lang_map):
    """Converts a language-to-family mapping into a member-to-family mapping."""
    out = {}

    for family in lang_map:
        for member in lang_map[family].members:
            out[member] = family

    return out


def get_submix_manifest(submix: str, split: str, cfg: DictConfig) -> List[str]:
    """
    Loads and processes manifests for all sources within a single submix.

    This function handles different types of manifest specifications, including
    standard file lists, clone specifications, and stem specifications. It also
    applies sampling probabilities based on language family if configured.
    """
    submix_cfg = cfg.data[submix]

    print("Submix: ", submix)

    lang_map = None
    lang_family_users = None
    if submix == "speech" and "lang" in cfg:
        lang_map = lang_to_family(cfg.lang)
        lang_users = {k: v.users for k, v in cfg.lang.items()}
        lang_family_users = lang_users

    manifest_dict = {}

    for subset_cfg in submix_cfg:
        subset_name = subset_cfg.subset

        if "clone" in subset_cfg:
            manifest_dict[subset_name] = CloneSpec(
                src=subset_cfg.clone.src,
                subset=subset_cfg.clone.subset,
                submix=subset_cfg.clone.submix,
            )
        elif "stem_src" in subset_cfg:
            manifest_dict[subset_name] = StemSpec(subset_cfg.stem_src, split, cfg)

        else:
            subset_sources = subset_cfg.src

            subset_manifest = []

            for src_name in subset_sources:
                manifest = get_subset_manifest(src_name=src_name, split=split, cfg=cfg)

                print(src_name, manifest)

                if manifest is None:
                    continue

                if submix == "speech":
                    if "langcode" not in manifest:
                        # TODO: fix this in the dataset organization
                        if "speech-faroese-slr125" in src_name:
                            manifest["langcode"] = "fao"
                        elif "speech-arabic-asc" in src_name:
                            manifest["langcode"] = "apc"
                        elif "speech-french-slr139" in src_name:
                            manifest["langcode"] = "fra"
                        else:
                            print(src_name)
                            raise ValueError("No langcode in manifest")
                    assert manifest["langcode"].isna().sum() == 0, print(
                        src_name, "\n", manifest
                    )
                subset_manifest.append(manifest)

            subset_manifest = pd.concat(subset_manifest).reset_index(drop=True)

            if submix == "speech":
                assert subset_manifest["langcode"].isna().sum() == 0

                if "speaker_gender" in subset_manifest:
                    subset_manifest["speaker_gender"] = subset_manifest[
                        "speaker_gender"
                    ].apply(clean_gender)

                subset_manifest["langcode"] = subset_manifest["langcode"].apply(
                    lambda x: x.lower().strip()
                )
                subset_manifest["language_family"] = subset_manifest["langcode"].apply(
                    lambda x: lang_map[x] if lang_map is not None else None
                )

                n_files_by_family = subset_manifest["language_family"].value_counts()
                users_per_file_by_family = {
                    k: max(1e6, lang_family_users[k]) / v
                    for k, v in n_files_by_family.items()
                }

                pprint(users_per_file_by_family)

                subset_manifest["sample_users"] = subset_manifest[
                    "language_family"
                ].apply(lambda x: users_per_file_by_family[x] if x is not None else 1.0)

                print(subset_manifest["sample_users"].describe())

                subset_manifest["sample_prob"] = (
                    subset_manifest["sample_users"]
                    / subset_manifest["sample_users"].sum()
                )
            else:
                print("!!")
                print(subset_manifest)
                print("!!")
                subset_manifest["sample_prob"] = 1.0 / len(subset_manifest)

            manifest_dict[subset_name] = subset_manifest

    return manifest_dict


def get_manifest(split: str, cfg: DictConfig) -> List[str]:
    """
    Loads all manifests for all submixes required for a given split.

    This is the main entry point for manifest loading. It iterates through all
    submixes defined in the data configuration and collects their manifests.

    Returns:
        A tuple containing:
        - A dictionary of all loaded manifests, nested by submix and subset.
        - A sorted list of all unique subset names found.
    """
    data_cfg = cfg.data

    manifest_dict = {}

    subsets = set()

    for submix in data_cfg:
        submix_manifest_dict = get_submix_manifest(submix=submix, split=split, cfg=cfg)
        manifest_dict[submix] = submix_manifest_dict

        submix_subsets = set(submix_manifest_dict.keys())

        subsets = subsets.union(submix_subsets)

    subsets = list(subsets)
    subsets = [subset for subset in subsets if subset != DEFAULT]

    subsets = sorted(subsets)

    if len(subsets) == 0:
        subsets = [DEFAULT]

    return manifest_dict, subsets
