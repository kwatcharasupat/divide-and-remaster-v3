FEMALE = "female"
MALE = "male"
UNSURE = "unsure"
UNKNOWN = "unknown"
FEMALE_UNSURE = "female_unsure"
MALE_UNSURE = "male_unsure"

GENDERS = {
    "Alexandra_Bogensperger": FEMALE,
    "Algy_Pug": MALE,
    "Friedrich": MALE,
    "AliceDe": FEMALE,
    "Anastasiia_Solokha": FEMALE,
    "Anka": FEMALE,
    "Anna_Samrowski": FEMALE,
    "Anna_Simon": FEMALE,
    "Anne": FEMALE,
    "Antoinette_Huting": FEMALE,
    "Anton": MALE,
    "Apneia": FEMALE,
    "Availle": FEMALE,
    "Bernd_Ungerer": MALE,
    "Boris": MALE,
    "Capybara": MALE,
    "Cate_Mackenzie": FEMALE,
    "Christian_Al-Kadi": MALE,
    "Christina_Lindgruen": FEMALE,
    "ClaudiaSterngucker": FEMALE,
    "ColOhr": MALE,
    "Crln_Yldz_Ksr": FEMALE,
    "DanielGrams": MALE,
    "DesirВe_LФffler": FEMALE,
    "Dini_Steyn": FEMALE,
    "Dirk_Weber": MALE,
    "DomBombadil": MALE,
    "Eki_Teebi": FEMALE,
    "Elli": FEMALE,
    "Eva_K": FEMALE,
    "Fabian_Grant": MALE,
    "Franziska_Paul": FEMALE,
    "Friedrich": MALE,
    "Frown": FEMALE,
    "Gaby": FEMALE,
    "Gesine": FEMALE,
    "Herman_Roskams": MALE,
    "Herr_Klugbeisser": MALE,
    "Hokuspokus": FEMALE,
    "Igor_Teaforay": MALE,
    "Imke_Grassl": FEMALE,
    "Ingo_Breuer": MALE,
    "IvanDean": MALE,
    "Jessi": FEMALE,
    "Joe_Kay": FEMALE,
    "Julia_Niedermaier": FEMALE,
    "Kaktus": FEMALE,
    "Kalynda": FEMALE,
    "Kanta": FEMALE,
    "Kara_Shallenberg": FEMALE,
    "KarinM": FEMALE,
    "Karlsson": MALE,
    "Klaus_Beutelspacher": MALE,
    "Klaus_Neubauer": MALE,
    "Knubbel": FEMALE,
    "Laila_Katinka": FEMALE,
    "Larry_Greene": MALE,
    "Lars_Rolander_(1942-2016)": MALE,
    "Lektor": MALE,
    "LillY": FEMALE,
    "LordOider": MALE,
    "LyricalWB": MALE,
    "Markus_Wachenheim": MALE,
    "Martin_Harbecke": MALE,
    "Mat": MALE,
    "Matze": MALE,
    "Monika_M._C": FEMALE,
    "Ohrbuch": MALE,
    "OldZach": MALE,
    "Orsina": FEMALE_UNSURE,  # ????????
    "PeWaOt": MALE,
    "Ragnar": MALE,
    "Rainer": MALE,
    "Ralf": MALE,
    "Ramona_Deininger-Schnabel": FEMALE,
    "Rebecca_Braunert-Plunkett": FEMALE,
    "RenateIngrid": FEMALE,
    "Rhigma": MALE,
    "Robert_Steiner": MALE,
    "Rogthey": MALE,
    "Sandra_Schmit": FEMALE,
    "Sascha": MALE,
    "Sebastian": MALE,
    "Sellafield": MALE,
    "Shanty": FEMALE,
    "Silke_Britz": FEMALE,
    "Silmaryllis": FEMALE,
    "Sonia": FEMALE,
    "Sonja": FEMALE,
    "Tabea": FEMALE,
    "Tanja_Ben_Jeroud": FEMALE,
    "Traxxo": MALE,
    "Ute2013": FEMALE,
    "Verena": FEMALE,
    "Victoria_Asztaller": FEMALE,
    "Wolfgang": MALE,
    "Zach_K": MALE_UNSURE,  # ????????
    "Zieraffe": MALE,
    "Zue_Von_Zob": FEMALE,
    "caromopfen": FEMALE,
    "danio": MALE,
    "ekyale": MALE,
    "fantaeiner": MALE,
    "fremdschaemen": FEMALE,
    "heeheekitty": FEMALE,
    "josimosi98": MALE,
    "keltoi": FEMALE,
    "leserchen": FEMALE,
    "lorda": MALE,
    "manuwolf": FEMALE,
    "marham63": MALE,
    "melaniesandra": FEMALE,
    "merendo07": MALE,
    "mindfulheart": FEMALE,
    "njall": MALE,
    "noonday": MALE,
    "schrm": MALE,
    "storylines": FEMALE_UNSURE,  # ??????
    "thinkofelephants": FEMALE,
}

import glob
import os
from itertools import cycle

import ffmpeg
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.contrib.concurrent import process_map

from ...const import AUDIO, MANIFEST
from ..utils import preface


def process_file(file, gender, split, cfg):
    subset = cfg.dataset.subset

    filename = os.path.basename(file)

    file_desc = file.split("hui-german")[-1].split("/")

    speaker = file_desc[0]
    book = file_desc[1]

    file = os.path.expandvars(file)

    output_path = os.path.join(
        cfg.data.cleaned_data_root, cfg.dataset.name, AUDIO, subset, split, filename
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    acodec = get_acodec_ffmpeg(cfg.audio.bit_depth)

    (
        ffmpeg.input(file)
        .output(
            output_path,
            format="wav",
            acodec=acodec,
            ar=cfg.audio.sampling_rate,
            ac=cfg.audio.channels,
            loglevel="error",
        )
        .run(overwrite_output=True)
    )

    metadata = {
        "file": file.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
        "cleaned_path": output_path.replace(
            cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
        ),
        "dsid_book_name": book,
        "dsid_speaker_id": speaker,
        "speaker_gender": gender,
        "subset": subset,
    }

    return metadata


def organize_split(manifest, split, cfg):
    df = manifest[manifest["split"] == split]

    files = df["file"].tolist()
    genders = df["gender"].tolist()

    metadata = process_map(
        process_file,
        files,
        genders,
        [split] * len(files),
        [cfg] * len(files),
        chunksize=1,
    )

    manifest = pd.DataFrame(metadata)

    manifest["langcode"] = "deu"
    manifest["language"] = "German"

    manifest["license"] = "CC-0"

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


def make_manifest(cfg):
    files = glob.glob(
        os.path.join("/root/data/cass-data/hui-german", "**", "wavs", "*.wav"),
        recursive=True,
    )

    df = pd.DataFrame(
        {
            "file": files,
        }
    )

    df["speaker"] = df["file"].apply(
        lambda x: os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(x))))
    )
    df["gender"] = df["speaker"].apply(lambda x: GENDERS[x].replace("_unsure", ""))
    df["_sinfo"] = df["file"].apply(lambda x: sf.info(x))
    df["duration"] = df["_sinfo"].apply(lambda x: x.frames / x.samplerate)
    del df["_sinfo"]

    df.to_csv(os.path.expandvars(cfg.path.manifest), index=False)


def get_gender_ratio(split_duration_by_gender, split_duration):
    ratio = {
        split: {
            g: split_duration_by_gender[split][g] / split_duration[split]
            if split_duration[split] > 0
            else 0
            for g in [MALE, FEMALE]
        }
        for split in split_duration
    }

    return ratio


def get_split_total_duration(split_duration_by_gender):
    total_duration = {
        split: sum(split_duration_by_gender[split].values())
        for split in split_duration_by_gender
    }

    return total_duration


def make_balanced_splits(speakers, cfg):
    seed_offset = 0

    splits = cfg.splits
    male_speakers = speakers.loc[MALE].copy()
    female_speakers = speakers.loc[FEMALE].copy()

    total_duration = speakers["duration"].sum()
    dataset_gender_ratio = {
        MALE: male_speakers["duration"].sum() / total_duration,
        FEMALE: female_speakers["duration"].sum() / total_duration,
    }

    while True:
        random_state = np.random.default_rng(cfg.seed + seed_offset)

        split_duration_by_gender = {
            split: {g: 0 for g in [MALE, FEMALE]} for split in splits
        }

        split_speakers = {split: [] for split in splits}

        target_duration = {
            split: total_duration * cfg.splits[split].duration_prop for split in splits
        }

        speakers_dict = {MALE: male_speakers, FEMALE: female_speakers}

        split_cycle = cycle(splits.keys())

        completed_splits = []

        split_duration = get_split_total_duration(split_duration_by_gender)

        split_gender_ratio = get_gender_ratio(split_duration_by_gender, split_duration)

        print(dataset_gender_ratio)

        make_round = False

        while True:
            if len(completed_splits) == len(splits):
                break

            split_diff = {
                split: max(
                    (
                        target_duration[split] * dataset_gender_ratio[MALE]
                        - split_duration_by_gender[split][MALE]
                    )
                    / (target_duration[split] * dataset_gender_ratio[MALE]),
                    (
                        target_duration[split] * dataset_gender_ratio[FEMALE]
                        - split_duration_by_gender[split][FEMALE]
                    )
                    / (target_duration[split] * dataset_gender_ratio[FEMALE]),
                )
                for split in splits
                if split not in completed_splits
            }

            probas = np.array([split_diff[s] for s in split_diff])
            probas = np.exp(probas)
            probas = probas / probas.sum()

            # print(probas)

            split = random_state.choice(list(split_diff.keys()), p=probas)

            # remaining_duration = {
            #     split: target_duration[split] - split_duration[split]
            #     for split in splits
            #     if split not in completed_splits
            # }

            # split = min(
            #     remaining_duration,
            #     key=remaining_duration.get
            # )

            if split in completed_splits:
                continue

            male_gender_ratio = split_gender_ratio[split][MALE]
            female_gender_ratio = split_gender_ratio[split][FEMALE]

            male_gender_ratio_diff = male_gender_ratio - dataset_gender_ratio[MALE]
            female_gender_ratio_diff = (
                female_gender_ratio - dataset_gender_ratio[FEMALE]
            )

            # print(f"Split: {split}")
            # print(f"Male diff: {male_gender_ratio_diff}")
            # print(f"Female diff: {female_gender_ratio_diff}")
            if len(speakers_dict[MALE]) == 0 and len(speakers_dict[FEMALE]) == 0:
                completed_splits.append(split)
                continue
            elif male_gender_ratio == 0 and female_gender_ratio == 0:
                next_gender = MALE
                current_ratio = -male_gender_ratio_diff
            elif len(speakers_dict[MALE]) > 0 and len(speakers_dict[FEMALE] > 0):
                # if abs(male_gender_ratio_diff) > 0.05 and abs(female_gender_ratio_diff) > 0.05:
                #     if female_gender_ratio_diff < 0:
                #         next_gender = FEMALE
                #         current_ratio = -female_gender_ratio_diff
                #         assert male_gender_ratio_diff > 0
                #     else:
                #         next_gender = MALE
                #         current_ratio = -male_gender_ratio_diff
                #         assert female_gender_ratio_diff >= 0
                if female_gender_ratio_diff < 0:
                    next_gender = FEMALE
                    current_ratio = -female_gender_ratio_diff
                else:
                    next_gender = MALE
                    current_ratio = -male_gender_ratio_diff

            elif len(speakers_dict[MALE]) == 0:
                next_gender = FEMALE
                current_ratio = -female_gender_ratio_diff
            elif len(speakers_dict[FEMALE]) == 0:
                next_gender = MALE
                current_ratio = -male_gender_ratio_diff
            else:
                raise ValueError("Should not reach here")

            next_speakers = speakers_dict[next_gender].copy()
            # print(next_gender, ":", len(next_speakers))

            remaining_duration = target_duration[split] - split_duration[split]
            needed_duration = remaining_duration * current_ratio
            if needed_duration < 0:
                # print("Split already has enough speakers")
                if len(completed_splits) < len(splits) and make_round < len(splits):
                    make_round += 1
                    continue
                else:
                    needed_duration = 0

            if len(next_speakers) == 0:
                completed_splits.append(split)
                continue

            next_speakers["diff"] = abs(next_speakers["duration"] - needed_duration)
            next_speakers = next_speakers.sort_values("diff")

            n_speakers = len(next_speakers)
            speaker = next_speakers.iloc[0]

            split_speakers[split].append(speaker.name)
            split_duration_by_gender[split][next_gender] += speaker["duration"]

            speakers_dict[next_gender] = speakers_dict[next_gender].drop(speaker.name)

            split_duration = get_split_total_duration(split_duration_by_gender)

            split_gender_ratio = get_gender_ratio(
                split_duration_by_gender, split_duration
            )

            # print(f"Split: {split}")
            # print(f"Gender ratio: {split_gender_ratio[split]}")

            if split_duration[split] >= target_duration[split]:
                completed_splits.append(split)
                continue

            if len(completed_splits) == len(splits):
                break

            if len(speakers_dict[MALE]) == 0 and len(speakers_dict[FEMALE]) == 0:
                break

        for split in splits:
            print(f"Split: {split}")
            print(f"Target Duration: {target_duration[split] / 3600}")
            print(f"Current Duration: {split_duration[split] / 3600 }")
            print(f"F/M Gender Ratio: {split_gender_ratio[split]}")

        print("Remaining speakers")
        for gender in speakers_dict:
            print(gender)
            print(speakers_dict[gender])

        split_ratio_offset = 0.15

        female_ratios = np.array(
            [split_gender_ratio[split][FEMALE] for split in splits]
        )

        if np.all(
            np.abs(female_ratios - dataset_gender_ratio[FEMALE]) < split_ratio_offset
        ):
            break

        seed_offset += 1

        print("Trying again with new seed offset")

    speaker_to_split = {}

    for split in split_speakers:
        for speaker in split_speakers[split]:
            speaker_to_split[speaker] = split

    return speaker_to_split


def organize_hui(cfg):
    preface(cfg)

    # print(cfg.path.manifest)

    if not os.path.exists(cfg.path.manifest):
        print("Making manifest")
        make_manifest(cfg)

    manifest = pd.read_csv(cfg.path.manifest)

    if "split" not in manifest.columns or cfg.remake_splits:
        print("Making splits")

        speakers = (
            manifest.groupby(["gender", "speaker"])
            .agg(
                {
                    "duration": "sum",
                    "file": "count",
                }
            )
            .sort_values("duration", ascending=False)
        )

        # print(speakers)

        speaker_to_split = make_balanced_splits(speakers, cfg)
        manifest["split"] = manifest["speaker"].apply(lambda x: speaker_to_split[x])
        manifest.to_csv(cfg.path.manifest, index=False)

    for split in cfg.splits:
        print(f"Organizing split: {split}")
        organize_split(manifest=manifest, split=split, cfg=cfg)
