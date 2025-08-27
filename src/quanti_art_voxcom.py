from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent))

import pandas as pd
import numpy as np
import logging
import importlib

from scipy.stats import pearsonr

from tqdm import tqdm

from voxcommunis.io import read_manifest
from utils import TqdmLoggingHandler

import argparse


mylogger = logging.getLogger(__name__)
mylogger.setLevel(logging.INFO)

handler = TqdmLoggingHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
mylogger.addHandler(handler)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",  # ex: ../../data/VoxCommunis/test-20h/arttts_pred/v6/grad_1000
    type=str,
)
parser.add_argument(
    "--save_dir",  # ex: ../../data/VoxCommunis/test-20h/analysis
    type=str,
)
parser.add_argument(
    "--sparc_dir",
    type=str,
    default="",  # ex: ../../data/VoxCommunis/encoded_audio_multi/it
)
parser.add_argument(
    "--manifest_path",  # ex: ../../data/VoxCommunis/test-20h/manifests/it.tsv
    type=str,
)
parser.add_argument(
    "--version",
    type=str,
    default="v6",
    help="Version of the model used to generate data",
)
parser.add_argument(
    "--params_name",
    type=str,
    default="params_v6",
    help="Parameters file name",
)
parser.add_argument(
    "--ckpt_name",
    type=str,
    default="grad_1000",
    help="Checkpoint name for the model used to generate data",
)

if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    manifest_path = Path(args.manifest_path)
    sparc_dir = Path(args.sparc_dir)
    params_name = args.params_name
    version = args.version
    ckpt_name = args.ckpt_name

    save_dir.mkdir(parents=True, exist_ok=True)
    params = importlib.import_module(f"configs.{params_name}")

    mylogger.info("Load filelist...")

    manifest = read_manifest(manifest_path)
    manifest_ids = set(manifest.keys())
    dir_files = list(data_dir.glob("*.npy"))
    dir_ids = set([f.stem for f in dir_files])
    file_ids = list(manifest_ids.intersection(dir_ids))

    mylogger.info("Found %d samples to process", len(file_ids))

    data = []
    with tqdm(
        range(len(file_ids)),
        total=len(file_ids),
        desc="Inference",
        position=1,
        dynamic_ncols=True,
    ) as progress_bar:
        for i in progress_bar:
            sample_id = file_ids[i]
            art = np.load(data_dir / f"{sample_id}.npy")
            assert len(art.shape) == 2, (
                f"Unexpected shape for articulatory data: {art.shape} != 2"
            )
            assert art.shape[0] == 29, (
                f"Unexpected shape for articulatory data: {art.shape[0]} != 29"
            )
            art = art[14:28, :].T  # take only decoder art (T,14)
            sparc_art = np.load(sparc_dir / f"emasrc/{sample_id}.npy")[
                :, :14
            ]  # (T, 14)
            # renormalize art using sparc stats
            pitch_mu, pitch_std = (
                sparc_art[:, 12].mean(),
                sparc_art[:, 12].std(),
            )
            art[:, 12] = art[:, 12] * pitch_std + pitch_mu  # denormalize pitch
            if params.log_normalize_loudness:
                loudness_mu, loudness_std = (
                    np.log(sparc_art[:, 13] + 1e-9).mean(),
                    np.log(sparc_art[:, 13] + 1e-9).std(),
                )
                art[:, 13] = (
                    art[:, 13] * loudness_std + loudness_mu
                )  # denormalize log-loudness
                art[:, 13] = np.exp(art[:, 13])  # delog log-loudness

            row = {
                "sample_id": sample_id,
            }

        pearson_ema, _ = pearsonr(art[:, :12], sparc_art[:, :12])
        pearson_pitch, _ = pearsonr(art[:, 12], sparc_art[:, 12])
        pearson_loudness, _ = pearsonr(art[:, 13], sparc_art[:, 13])
        row.update(
            {
                "pcc_ema": np.mean(pearson_ema),
                "pcc_pitch": pearson_pitch,
                "pcc_loudness": pearson_loudness,
            }
        )
        data.append(row)

    res_df = pd.DataFrame(data)
    res_df.to_csv(save_dir / f"quanti_art_comp_{version}_{ckpt_name}.csv", index=False)
