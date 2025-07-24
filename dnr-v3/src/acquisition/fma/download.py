import os

from tqdm.contrib.concurrent import process_map

from ..common import download_file

FMA_URL = {
    "metadata": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
    "small": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    "medium": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
    "large": "https://os.unil.cloud.switch.ch/fma/fma_large.zip",
    "full": "https://os.unil.cloud.switch.ch/fma/fma_full.zip",
}


def download_subset(subset, data_root=os.path.expandvars("$RAW_DATA_ROOT")):
    """Downloads a single subset of the FMA dataset."""
    url = FMA_URL[subset]

    output = f"{data_root}/fma"

    os.makedirs(output, exist_ok=True)

    output_file = f"{output}/{os.path.basename(url)}"

    download_file(url, output_file=output_file)


def download(
    subsets=["medium", "large", "metadata"],
    data_root=os.path.expandvars("$RAW_DATA_ROOT"),
):
    """Downloads multiple subsets of the FMA dataset in parallel."""
    process_map(download_subset, subsets, [data_root] * len(subsets), chunksize=1)


if __name__ == "__main__":
    import fire

    fire.Fire()
