import glob
import os
from pprint import pprint

import pandas as pd
from tqdm.contrib.concurrent import process_map

OPENSLR_URL = "https://us.openslr.org/resources"


from ..common import download_file


def download(
    slr_id,
    alias,
    data_root=os.path.expandvars("$RAW_DATA_ROOT"),
    ignore_existing=False,
    y=True,
    download_checksum=False,
):
    """
    Downloads all files for a given OpenSLR resource ID.

    This function scrapes the resource page to find all downloadable files
    and then downloads them in parallel.
    """
    url = f"{OPENSLR_URL}/{slr_id}"

    output = f"{data_root}/slr{slr_id}-{alias}"

    os.makedirs(output, exist_ok=True)

    df = pd.read_html(url)[0]

    df = df.dropna(subset="Name")

    df = df[df["Name"] != "Parent Directory"]
    df = df[df["Name"] != "about.html"]

    if not download_checksum:
        df = df[df["Name"] != "checksum.md5"]

    files = df["Name"].tolist()

    output_files = [f"{output}/{file}" for file in files]
    urls = [f"{url}/{file}" for file in files]

    if ignore_existing:
        urls_ = []
        output_files_ = []

        for url, output_file in zip(urls, output_files):
            if not os.path.exists(output_file):
                urls_.append(url)
                output_files_.append(output_file)

        urls = urls_
        output_files = output_files_

    if len(output_files) == 0:
        print("No files to download")
        return

    print(f"Downloading {len(output_files)} files:")
    pprint(output_files)

    if y or input("Proceed? [y/n]") == "y":
        process_map(download_file, urls, output_files, chunksize=1)


def check_license(data_root=os.path.expandvars("$RAW_DATA_ROOT"), prefix="slr"):
    """Scans downloaded SLR directories and prints the first line of any LICENSE file."""
    licenses = glob.glob(f"{data_root}/{prefix}*/LICENSE*", recursive=True)

    for l in licenses:
        first_line = open(l).readline()

        print(l, first_line)


if __name__ == "__main__":
    import fire

    fire.Fire()
