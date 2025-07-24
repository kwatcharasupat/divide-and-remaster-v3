import ast
import os
import re

import ffmpeg
import librosa
import numpy as np
import pandas as pd
import torch
from aa_smad_cli.predictor import crnn_predictor as smad_predictor
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ...const import AUDIO, MANIFEST
from ..smad import get_segments_from_activation
from ..utils import soundfile_export
from ..utils.ffmpeg import ffmpeg_get_acodec, ffmpeg_reformat_to_buffer


def organize_file(
    path,
    tmp_path,
    title,
    speaker,
    gender,
    license_usage,
    license_name,
    split,
    subset,
    cfg,
):
    if not os.path.exists(tmp_path):
        raise FileNotFoundError(f"File not found: {tmp_path}")

    # audio = np.ascontiguousarray(audio)

    old_basename = os.path.basename(path).replace(".mp3", "")
    old_basename = re.sub(r"\W+", "", old_basename)

    while "__" in old_basename:
        old_basename = old_basename.replace("__", "_")

    subset = subset.format(
        license=license_usage,
    )

    temp_manifest = os.path.join(
        cfg.data.cleaned_data_root,
        cfg.dataset.name,
        "temp_" + MANIFEST,
        subset,
        split,
        f"{old_basename}.csv",
    )

    if os.path.exists(temp_manifest):
        return pd.read_csv(temp_manifest)

    smad_path = path.replace(".mp3", ".csv").replace(
        cfg.path.audio_path, cfg.path.smad_path
    )

    if not os.path.exists(smad_path):
        smad_path = smad_path.replace(
            cfg.data.raw_data_root, "/fsx_vfx/projects/aa-cass/data/cass-data-raw"
        )

        if not os.path.exists(smad_path):
            print(f"SMAD file not found: {smad_path}")
            return []

    smad = pd.read_csv(smad_path)
    smad = smad[smad["class"] == "vox-only"]

    smad["duration"] = smad["end_time"] - smad["start_time"]

    smad = smad[smad["duration"] >= cfg.segmentation.min_segment_duration_seconds]

    segment_idx = 0

    manifest = []

    audio = ffmpeg_reformat_to_buffer(
        path,
        sampling_rate=cfg.audio.sampling_rate,
        num_channels=cfg.audio.channels,
    ).T

    for _, row in tqdm(smad.iterrows(), total=len(smad)):
        start_time = row["start_time"]
        end_time = row["end_time"]

        start_samples = int(start_time * cfg.audio.sampling_rate)
        end_samples = int(end_time * cfg.audio.sampling_rate)
        duration = end_time - start_time

        segment = audio[:, start_samples:end_samples]

        if duration > cfg.segmentation.max_segment_duration_seconds:
            # print("Segment too long, splitting")
            # continue

            for threshold in [60, 40, 20, 10]:
                boundaries = librosa.effects.split(
                    segment, top_db=threshold, frame_length=4096, hop_length=1024
                )

                durations = boundaries[:, 1] - boundaries[:, 0]
                if np.any(
                    durations
                    > cfg.segmentation.max_segment_duration_seconds
                    * cfg.audio.sampling_rate
                ):
                    break

            starts = boundaries[:, 0]
            ends = boundaries[:, 1]

            for _, (start, end) in enumerate(zip(starts, ends)):
                if (
                    end - start
                    < cfg.segmentation.min_segment_duration_seconds
                    * cfg.audio.sampling_rate
                ):
                    continue

                if (
                    end - start
                    > cfg.segmentation.max_segment_duration_seconds
                    * cfg.audio.sampling_rate
                ):
                    # print("Segment too long, continuing anyway")
                    pass

                subsegment = segment[:, start:end]

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
                        "file": path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
                        "cleaned_path": segment_path.replace(
                            cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
                        ),
                        "title": title,
                        "speaker": speaker,
                        "gender": gender,
                        "license": license_name,
                        "license_usage": license_usage,
                        "subset": subset,
                        "split": split,
                        "segment_idx": segment_idx,
                        "start_time": start / cfg.audio.sampling_rate + start_time,
                        "end_time": end / cfg.audio.sampling_rate + start_time,
                    }
                )

                segment_idx += 1

        else:
            segment_path = os.path.join(
                cfg.data.cleaned_data_root,
                cfg.dataset.name,
                AUDIO,
                subset,
                split,
                f"{old_basename}_{segment_idx:05d}.wav",
            )

            try:
                soundfile_export(segment.T, segment_path, cfg)
            except Exception as e:
                print(f"Error for track: {path}")
                print(e)
                continue

            manifest.append(
                {
                    "file": path.replace(cfg.data.raw_data_root, "$RAW_DATA_ROOT"),
                    "cleaned_path": segment_path.replace(
                        cfg.data.cleaned_data_root, "$CLEANED_DATA_ROOT"
                    ),
                    "title": title,
                    "speaker": speaker,
                    "gender": gender,
                    "license": license_name,
                    "license_usage": license_usage,
                    "subset": subset,
                    "split": split,
                    "segment_idx": segment_idx,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

            segment_idx += 1

    df = pd.DataFrame(manifest)

    os.makedirs(os.path.dirname(temp_manifest), exist_ok=True)

    df.to_csv(temp_manifest, index=False)

    return df


def check_smad_exists(path, cfg):
    filename = path.replace(".mp3", ".csv").replace(
        cfg.path.tmp_audio_path, cfg.path.smad_path
    )

    if not os.path.exists(filename):
        # print(f"SMAD file not found: {filename}")
        return path

    return None


def run_smad_on_files(paths, cfg):
    # n_groups = 4

    # n_tracks = len(paths)

    # n_tracks_per_group = n_tracks // n_groups

    # track_groups = [
    #     paths[i : i + n_tracks_per_group]
    #     for i in range(0, n_tracks, n_tracks_per_group)
    # ]

    # process_map(
    #     _run_smad_on_files,
    #     track_groups,
    #     [cfg] * len(track_groups),
    #     range(len(track_groups)),
    #     chunksize=1,
    #     max_workers=n_groups
    # )

    print("Running SMAD on files")

    _run_smad_on_files(paths, cfg)


def get_smad_activations(
    src_path,
    smad,
    cfg,
    overwrite=False,
):
    smad_path = src_path.replace(".wav", ".csv").replace(
        cfg.path.tmp_audio_path, cfg.path.smad_path
    )

    if not overwrite:
        if os.path.exists(smad_path):
            return

    try:
        ffmpeg.probe(src_path)
    except ffmpeg.Error as e:
        print(f"Error for track: {src_path}")
        print(e)
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

    print(f"Saved SMAD activations to {smad_path}")


def _run_smad_on_files(
    paths,
    cfg,
    position=0,
    overwrite=False,
):
    smad = smad_predictor(cuda=torch.cuda.is_available())

    for path in tqdm(paths, position=position):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        try:
            get_smad_activations(path, smad, cfg, overwrite=overwrite)
        except Exception as e:
            print(f"Error for track: {path}")
            print(e)

            try:
                ffmpeg.probe(path)
            except ffmpeg.Error as e:
                print(f"Error for track: {path}")
                print(e)

                os.remove(path)

            continue

    del smad


def check_tmp_wav_exists(path, cfg):
    cleaned_path = (
        path.replace(".mp3", ".wav")
        .replace(cfg.path.audio_path, cfg.path.tmp_audio_path)
        .replace(cfg.data.raw_data_root, "/fsx_vfx/projects/aa-cass/data/cass-data-raw")
    )

    if not os.path.exists(cleaned_path):
        return path

    return None


def make_tmp_wav(path, cfg):
    cleaned_path = (
        path.replace(".mp3", ".wav")
        .replace(cfg.path.audio_path, cfg.path.tmp_audio_path)
        .replace(cfg.data.raw_data_root, "/fsx_vfx/projects/aa-cass/data/cass-data-raw")
    )

    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)

    try:
        (
            ffmpeg.input(path)
            .output(
                cleaned_path,
                acodec="pcm_s16le",
                ar=16000,  # cfg.audio.sampling_rate,
                ac=1,
                loglevel="error",
            )
            .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        print(f"Error for track: {path}")
        print(e)
        os.remove(cleaned_path)
        return


def make_formatted_tmp_wav(path, cfg):
    cleaned_path = (
        path.replace(".mp3", ".wav")
        .replace(cfg.path.audio_path, cfg.path.tmp_audio_path)
        .replace(cfg.data.raw_data_root, "/fsx_vfx/projects/aa-cass/data/cass-data-raw")
    )

    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)

    acodec = ffmpeg_get_acodec(cfg.audio.bit_depth)

    try:
        (
            ffmpeg.input(path)
            .output(
                cleaned_path,
                acodec=acodec,
                ar=cfg.audio.sampling_rate,
                ac=cfg.audio.channels,
                loglevel="error",
            )
            .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        print(f"Error for track: {path}")
        print(e)
        os.remove(cleaned_path)
        return


def get_mp3_quality(path):
    try:
        from mutagen.mp3 import MP3

        audio = MP3(path)
        return np.round(audio.info.bitrate / 1000)
    except Exception as e:
        print(e)
        return -1


def organize_split(license_manifest, split, cfg):
    json_path = os.path.join(cfg.data.raw_data_root, cfg.splits[split].manifest)

    df = pd.read_json(json_path, orient="index").reset_index(names=["id"])

    df["folder"] = df["path"].apply(lambda x: x.replace("../wavs/", "").split("/")[0])
    df = df.merge(
        license_manifest[["Title", "folder", "license_usage", "cleaned_license"]],
        on="folder",
        how="left",
    )

    n_missing = df["license_usage"].isna().sum()
    print(f"Missing {n_missing} licenses")

    if n_missing > 0:
        raise ValueError("Missing licenses")

    print("Total files:", len(df))

    print(
        df.value_counts(
            [
                "license_usage",
                "cleaned_license",
            ]
        )
    )

    if cfg.audiocite.license_filter:
        df = df[df["license_usage"].isin(cfg.audiocite.license_filter)]
        print("Total files after filtering:", len(df))

    df["path"] = df["path"].apply(
        lambda x: x.replace(
            "../wavs", os.path.join(cfg.data.raw_data_root, cfg.path.audio_path)
        )
    )

    # if False:
    if cfg.audiocite.check_missing_temp_wavs:
        print("Checking for missing temporary wav files")

        overwrite = False

        if overwrite:
            missing_temp_wavs = df["path"].tolist()
        else:
            missing_temp_wavs = process_map(
                check_tmp_wav_exists,
                df["path"].tolist(),
                [cfg] * len(df),
                chunksize=8,
            )

        missing_temp_wavs = list(set([x for x in missing_temp_wavs if x is not None]))

        df_missing_tmp = df[df["path"].isin(missing_temp_wavs)].sort_values("duration")

        missing_temp_wavs = df_missing_tmp["path"].tolist()

        print("total tmp missing", len(df_missing_tmp))

        process_map(
            make_tmp_wav,
            missing_temp_wavs,
            [cfg] * len(df),
            chunksize=1,
        )

    if cfg.audiocite.run_smad:
        df["_path"] = df["path"].apply(
            lambda x: x.replace(".mp3", ".wav")
            .replace(cfg.path.audio_path, cfg.path.tmp_audio_path)
            .replace(
                cfg.data.raw_data_root, "/fsx_vfx/projects/aa-cass/data/cass-data-raw"
            )
        )
        paths = df.sort_values("duration")["_path"].tolist()
        print("Checking for missing SMAD files")

        missing_smads = process_map(
            check_smad_exists, paths, [cfg] * len(paths), chunksize=1, max_workers=16
        )

        missing_smads = [x for x in missing_smads if x is not None]

        df_missing = df[df["_path"].isin(missing_smads)].sort_values("duration")
        print("total smad missing", len(df_missing))

        # only run on files < 5 min for now
        df_missing5 = df_missing[df_missing["duration"] < 60 * 5]
        print("total smad missing < 5min", len(df_missing5))

        missing_smads = df_missing["_path"].tolist()

        if len(missing_smads) > 0:
            print("Total SMAD missing:", len(missing_smads))
            run_smad_on_files(missing_smads, cfg)
    else:
        pass

    if cfg.audiocite.run_formatted_temp_wavs:
        process_map(
            make_formatted_tmp_wav,
            df["path"].tolist(),
            [cfg] * len(df),
            chunksize=1,
        )

    mp3_qualities = process_map(
        get_mp3_quality,
        df["path"].tolist(),
        chunksize=1,
    )

    df["mp3_quality"] = mp3_qualities
    df = df[df["mp3_quality"] >= 64]

    df["tmp_path"] = df["path"].apply(
        lambda x: x.replace(".mp3", ".wav")
        .replace(cfg.path.audio_path, cfg.path.tmp_audio_path)
        .replace(cfg.data.raw_data_root, "/fsx_vfx/projects/aa-cass/data/cass-data-raw")
    )

    df = df.sort_values("duration")

    paths = df["path"].tolist()
    tmp_paths = df["tmp_path"].tolist()
    speakers = df["spk_id"].tolist()
    genders = df["spk_gender"].tolist()
    license_usages = df["license_usage"].tolist()
    license_names = df["cleaned_license"].tolist()
    titles = df["Title"].tolist()

    subset = cfg.dataset.subset

    manifests = process_map(
        organize_file,
        paths,
        tmp_paths,
        titles,
        speakers,
        genders,
        license_usages,
        license_names,
        [split] * len(paths),
        [subset] * len(paths),
        [cfg] * len(paths),
        chunksize=1,
        # max_workers=1
    )

    manifest_df = pd.concat([m for m in manifests if len(m) > 0])
    manifest_df["split"] = split

    if len(cfg.audiocite.license_filter) == 1:
        subset = subset.format(license=cfg.audiocite.license_filter[0])
    else:
        raise NotImplementedError("Multiple licenses not supported")

    manifest_path = os.path.join(
        cfg.data.cleaned_data_root, cfg.dataset.name, MANIFEST, subset, f"{split}.csv"
    )

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    manifest_df.to_csv(manifest_path, index=False)


def _get_cc_type(attribution, commercial, derivative, share_alike):
    if attribution is None:
        return "Unclear CC License"

    cc_type = ["CC"]

    if attribution:
        cc_type.append("BY")

    if not commercial:
        cc_type.append("NC")

    if not derivative:
        cc_type.append("ND")

    if share_alike:
        cc_type.append("SA")

    return "-".join(cc_type)


def clean_license(licenses):
    if (
        licenses
        == "[('http://creativecommons.fr/licences/', 'Creative Commons BY (attribution) NC (Pas d'utilisation commerciale) ND (Pas de modification)')]"
    ):
        licenses = "[('http://creativecommons.fr/licences/', 'Creative Commons BY (attribution) NC (Pas d`utilisation commerciale) ND (Pas de modification)')]"
    elif (
        licenses
        == "[('http://creativecommons.fr/licences/', 'Creative Commons BY (attribution) NC (Pas d'utilisation commerciale) SA (Partage dans les mêmes conditions)')]"
    ):
        licenses = "[('http://creativecommons.fr/licences/', 'Creative Commons BY (attribution) NC (Pas d`utilisation commerciale) SA (Partage dans les mêmes conditions)')]"
    elif (
        licenses
        == "[('http://creativecommons.fr/licences/', 'Creative Commons BY (attribution) NC (Pas d'utilisation commerciale)')]"
    ):
        licenses = "[('http://creativecommons.fr/licences/', 'Creative Commons BY (attribution) NC (Pas d`utilisation commerciale)')]"
    elif (
        licenses
        == "[('http://artlibre.org/', 'Art Libre'), ('http://creativecommons.fr/licences/' Creative Commons BY (attribution) NC (Pas d'utilisation commerciale) ND (Pas de modification)')]"
    ):
        licenses = "[('http://artlibre.org/', 'Art Libre'), ('http://creativecommons.fr/licences/', 'Creative Commons BY (attribution) NC (Pas d`utilisation commerciale) ND (Pas de modification)')]"

    licenses = ast.literal_eval(licenses)

    license_types = []
    attributions = []
    commercials = []
    derivatives = []
    share_alikes = []

    for url, license_name in licenses:
        if "Creative Commons" in license_name:
            if "BY" in license_name:
                attribution = True
            else:
                attribution = None

            if "NC" in license_name:
                commercial = False
            else:
                commercial = True

            if "ND" in license_name:
                derivative = False
            else:
                derivative = True

            if "SA" in license_name:
                share_alike = True
            else:
                share_alike = False

            license_type = _get_cc_type(
                attribution, commercial, derivative, share_alike
            )

        elif "Art Libre" in license_name:
            license_type = "Art Libre"

            attribution = True
            commercial = True
            derivative = True
            share_alike = True

        else:
            raise NotImplementedError(f"Unknown license: {license_name}")

        license_types.append(license_type)
        attributions.append(attribution)
        commercials.append(commercial)
        derivatives.append(derivative)
        share_alikes.append(share_alike)

    if "Unclear CC License" in license_types:
        return (
            "Unclear License",
            True,
            False,
            False,
            True,
        )  # use the most restrictive license

    license_ = "; ".join(license_types)
    attribution = any(attributions)  # if any is true, then it is true
    commercial = all(commercials)  # if all is true, then it is true
    derivative = all(derivatives)  # if all is true, then it is true
    share_alike = any(share_alikes)  # if any is true, then it is true

    return license_, attribution, commercial, derivative, share_alike


def load_license_manifest(cfg):
    path = os.path.join(cfg.data.raw_data_root, cfg.path.license_manifest)

    df = pd.read_csv(path, sep=";")

    (
        df["cleaned_license"],
        df["attribution"],
        df["commercial"],
        df["derivative"],
        df["share_alike"],
    ) = zip(*df["License"].apply(clean_license))

    print(
        df.value_counts(
            [
                "derivative",
                "commercial",
            ]
        ).sort_index()
    )

    df["license_usage"] = df[["commercial", "derivative"]].apply(
        lambda x: "nd"
        if not x["derivative"]
        else "nc"
        if not x["commercial"]
        else "com",
        axis=1,
    )
    df["folder"] = df["Download folder"].apply(lambda x: x.rstrip("/").split("/")[-1])

    df = df[["Title", "Speaker", "license_usage", "cleaned_license", "folder"]]

    return df


def check_sampling_rate(path):
    try:
        info = ffmpeg.probe(os.path.expandvars(path))
        return {
            "file": path,
            "sampling_rate": int(info["streams"][0]["sample_rate"]),
        }
    except Exception as e:
        print(e)
        return -1


def organize_audiocite(cfg):
    # preface(cfg)

    # license_manifest = load_license_manifest(cfg)

    # for split in cfg.splits:
    #     print(f"Organizing split: {split}")
    #     organize_split(license_manifest=license_manifest, split=split, cfg=cfg)

    # for split in cfg.splits:
    #     manifest_path = os.path.join(
    #         cfg.data.cleaned_data_root,
    #         cfg.dataset.name,
    #         MANIFEST,
    #         cfg.dataset.subset.format(license=cfg.audiocite.license_filter[0]),
    #         f"{split}.csv"
    #     )

    #     manifest = pd.read_csv(manifest_path)

    #     mp3_files = manifest["file"].unique()

    #     sampling_rates = process_map(
    #         check_sampling_rate,
    #         mp3_files,
    #         chunksize=1,
    #     )

    #     df_fs = pd.DataFrame(sampling_rates)

    #     manifest = manifest.merge(df_fs, on="file")

    #     to_remove = manifest[manifest["sampling_rate"] < 44100]["cleaned_path"].tolist()

    #     print("Removing", len(to_remove), "files")

    #     for path in to_remove:
    #         os.remove(os.path.expandvars(path))

    #     manifest = manifest[~manifest["cleaned_path"].isin(to_remove)]

    #     manifest.to_csv(manifest_path, index=False)

    all_manifests = []

    for split in cfg.splits:
        manifest_path = os.path.join(
            cfg.data.cleaned_data_root,
            cfg.dataset.name,
            MANIFEST,
            cfg.dataset.subset.format(license=cfg.audiocite.license_filter[0]),
            f"{split}.csv",
        )

        manifest = pd.read_csv(manifest_path)

        all_manifests.append(manifest)

    all_manifest = pd.concat(all_manifests)

    print(all_manifest.columns)

    print(all_manifest[["speaker", "gender"]].drop_duplicates().value_counts("gender"))
