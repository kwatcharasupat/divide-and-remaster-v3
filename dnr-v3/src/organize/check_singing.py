import glob
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from demucs.apply import apply_model
from demucs.pretrained import get_model
from tqdm import tqdm


def run_demucs(wav, model, cuda_idx):
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std()

    wav = wav.repeat(2, 1)

    sources = apply_model(
        model,
        wav[None, ...],
        device=f"cuda:{cuda_idx}",
        shifts=0,
        split=True,
        overlap=0,
    )[0]

    sources *= ref.std()
    sources += ref.mean()

    sources = {name: sources[i, :] for i, name in enumerate(model.sources)}

    return sources


def dbrms(audio):
    return 10 * torch.log10(torch.mean(torch.square(audio)) + 1e-8)


def check_singing_file(file: str, separator, cuda_idx):
    audio = np.load(
        os.path.expandvars(file).replace(".wav", ".npy").replace("audio", "npy32")
    )

    audio = torch.from_numpy(audio).cuda(cuda_idx)
    # print(audio.device)
    separated = run_demucs(audio, separator, cuda_idx=cuda_idx)

    vox = separated["vocals"]
    novox = sum([separated[key] for key in separated.keys() if key != "vocals"])

    vox_energy = dbrms(vox)
    novox_energy = dbrms(novox)

    vox_to_novox = vox_energy - novox_energy

    out_dict = {
        "file": file,
        "vox_energy": float(vox_energy.cpu().numpy()),
        "novox_energy": float(novox_energy.cpu().numpy()),
        "vox_to_novox": float(vox_to_novox.cpu().numpy()),
    }

    df = pd.DataFrame.from_records([out_dict])

    out_path = (
        os.path.expandvars(file).replace("audio", "singcheck").replace(".wav", ".csv")
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df.to_csv(out_path, index=False)


def infer(rank, queue, pbqueue, n_gpus=4):
    cuda_idx = rank % n_gpus

    print(f"Rank {rank} using cuda {cuda_idx}")

    separator = get_model("htdemucs").cuda(cuda_idx)  # .to(f"cuda:{cuda_idx}")

    for submodel in separator.models:
        submodel.cuda(cuda_idx)

    while True:
        file = queue.get()
        if file is None:  # check for sentinel value
            break

        out = check_singing_file(file, separator, cuda_idx)

        pbqueue.put_nowait(1)

        if queue.empty():
            break

    print(f"Rank {rank} completed")


def update_bar(pbqueue, total):
    pbar = tqdm(total=total)

    while True:
        x = pbqueue.get()
        pbar.update(1)

        if x is None:
            break

    pbar.close()


def check_singing_files(files: List[str], n_gpus=4, n_workers=32):
    max_workers = n_workers

    queue = mp.Queue()
    pbqueue = mp.Queue()

    processes = []

    bar_process = mp.Process(target=update_bar, args=(pbqueue, len(files)), daemon=True)
    bar_process.start()

    for rank in range(max_workers):
        p = mp.Process(target=infer, args=(rank, queue, pbqueue, n_gpus))
        p.start()
        processes.append(p)
    for file in tqdm(files):
        queue.put(file)
    for _ in range(max_workers):
        queue.put(None)  # sentinel value to signal subprocesses to exit
        queue.put(None)  # sentinel value to signal subprocesses to exit
        queue.put(None)  # sentinel value to signal subprocesses to exit
    for p in processes:
        p.join()  # wait for all subprocesses to finish

    pbqueue.put(None)


def check_singing(
    manifest_root: str,
    manifest_format: str = "**/44k/{split}.csv",
    splits: List[str] = ["train", "val", "test"],
    n_gpus=4,
    n_workers=32,
):
    for split in splits:
        print(f"Running split: {split}")

        manifest_path = os.path.join(manifest_root, manifest_format.format(split=split))
        manifests = glob.glob(manifest_path, recursive=True)

        print(f"Found {len(manifests)} manifests")

        for manifest in manifests:
            print(f"Checking {manifest}")

            df = pd.read_csv(manifest)
            gfiles = df.cleaned_path.tolist()

            # df["group"] = df.cleaned_path.apply(lambda x: os.path.basename(x)[:3])

            # for group, dfg in df.groupby("group"):

            gfiles = df.cleaned_path.tolist()

            out_dicts = check_singing_files(
                gfiles, n_gpus=n_gpus, n_workers=min(n_workers, len(gfiles))
            )


if __name__ == "__main__":
    import fire

    fire.Fire(check_singing)
