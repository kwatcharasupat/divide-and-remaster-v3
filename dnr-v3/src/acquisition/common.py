import glob
import os

import pandas as pd
import requests
import soundfile as sf
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def get_audio_specs(
    prefix,
    data_root=os.path.expandvars("$RAW_DATA_ROOT"),
    usable_threshold=44100,
    glob_pattern="**",
    extensions=[".wav", ".flac", ".mp3", ".opus", ".ogg"],
):
    """Inspects all audio files in a directory and prints a summary of their specs."""
    folders = sorted(os.listdir(data_root))
    folders = [
        f"{data_root}/{f}"
        for f in folders
        if os.path.isdir(f"{data_root}/{f}") and f.startswith(prefix)
    ]

    folders = sorted(folders)

    for folder in folders:
        try:
            print("inspecting", folder)

            audio_files = []

            for ext in extensions:
                audio_files += glob.glob(
                    f"{folder}/{glob_pattern}/*{ext}", recursive=True
                )

            metadata = process_map(inspect_audio, audio_files, chunksize=16)

            df = pd.DataFrame(metadata)

            df.to_csv(f"{folder}/audio_metadata.csv", index=False)

            fs_stat = df["sampling_rate"].value_counts()

            format_stat = df[["format", "subtype"]].value_counts()

            channel_stat = df["channels"].value_counts()

            print("Sampling Rate:")
            print(fs_stat)

            print("Format:")
            print(format_stat)

            print("Channels:")
            print(channel_stat)

            total_hours = (df["frames"] / df["sampling_rate"]).sum() / 3600

            print(f"Total duration: {total_hours:.1f} hours")

            df_usable = df[df["sampling_rate"] >= usable_threshold]
            total_hours = (
                df_usable["frames"] / df_usable["sampling_rate"]
            ).sum() / 3600

            print(f"Total duration of usable audio: {total_hours:.1f} hours")

            format_stat_usable = df_usable[["format", "subtype"]].value_counts()

            print("Format of usable audio:")
            print(format_stat_usable)

            print("===============================================")
        except Exception as e:
            print(e)
            continue


def extract_archive(file, folder):
    """Extracts a single archive file (tar or zip)."""
    if os.path.isdir(f"{folder}/{file}"):
        extract_folder(f"{folder}/{file}")
        return

    if ".tar" in file or ".tgz" in file:
        print(f"Extracting {file}")

        os.system(f"tar --overwrite -xvf {folder}/{file} -C {folder}")

    if ".zip" in file:
        print(f"Extracting {file}")

        os.system(f"unzip -o {folder}/{file} -d {folder}")


def extract_folder(folder, parallel=True):
    """Extracts all archives in a given folder."""
    files = os.listdir(folder)

    if parallel:
        process_map(extract_archive, files, [folder] * len(files), chunksize=1)
    else:
        for file in files:
            extract_archive(file, folder)


def extract(prefix, data_root=os.path.expandvars("$RAW_DATA_ROOT"), parallel=True):
    """Finds and extracts all archives in folders matching a prefix."""
    folders = os.listdir(data_root)
    folders = [
        f"{data_root}/{f}"
        for f in folders
        if os.path.isdir(f"{data_root}/{f}") and f.startswith(prefix)
    ]

    print(folders)

    for folder in folders:
        extract_folder(folder, parallel=parallel)


def download_file(url, output_file):
    """Downloads a file from a URL to a specified output path with a progress bar."""
    print(f"Downloading {url} to {output_file}")

    res = requests.get(url, stream=True)

    total_size = int(res.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(output_file, "wb") as file:
            for data in res.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)


def inspect_audio(audio_file):
    """Reads the metadata of an audio file using soundfile."""
    try:
        info = sf.info(audio_file)
        return {
            "audio_path": audio_file,
            "sampling_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype,
            "extra": info.extra_info,
        }
    except Exception as e:
        print(e)

        return {
            "audio_path": audio_file,
            "sampling_rate": None,
            "channels": None,
            "frames": None,
            "format": None,
            "subtype": None,
            "extra": None,
        }


if __name__ == "__main__":
    import fire

    fire.Fire()
