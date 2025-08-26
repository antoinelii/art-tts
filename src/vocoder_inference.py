import numpy as np
import logging
import json
from tqdm import tqdm
from pathlib import Path
from scipy.io.wavfile import write

import torch

from utils import TqdmLoggingHandler, parse_filelist

import sys

sys.path.append("./hifi-gan/")
from env import AttrDict
from models import Generator as HiFiGAN

import argparse

HIFIGAN_CONFIG = "./checkpts/hifigan-config.json"
HIFIGAN_CHECKPT = "./checkpts/hifigan.pt"

mel_versions = ["v2", "v3", "v2_full"]

mylogger = logging.getLogger(__name__)
mylogger.setLevel(logging.INFO)

handler = TqdmLoggingHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
mylogger.addHandler(handler)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",  # ex: ../../data/LJSpeech-1.1/arttts_pred/v2/grad_200
    type=str,
)
parser.add_argument(
    "--save_dir",  # ex: ../../data/LJSpeech-1.1/hifigan_pred/v2/grad_200
    type=str,
)
parser.add_argument(
    "--filelist_path",  # ex resources/filelists/ljspeech/valid_v2.txt
    type=str,
)
parser.add_argument(
    "--main_dir",  # ex: ../../data/
    type=str,
)
parser.add_argument(
    "--version",
    type=str,  # bool
)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--src_mel",  # if from model encoder or decoder
    type=str,
    default="",  # , choices=["encoder", "decoder"],
)


if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    filelist_path = Path(args.filelist_path)
    main_dir = Path(args.main_dir)
    device = args.device
    src_mel = args.src_mel
    version = args.version

    save_dir.mkdir(parents=True, exist_ok=True)

    mylogger.info("Initializing HiFi-GAN...")

    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(
        torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)["generator"]
    )
    _ = vocoder.to(device).eval()
    vocoder.remove_weight_norm()

    mylogger.info("Load filelist...")

    filepaths_list = parse_filelist(filelist_path)
    filenames_list = [fp[0].split("/")[-1] for fp in filepaths_list]

    mylogger.info("Filelist loaded with %d samples", len(filenames_list))

    # if from arttts_pred need to extract the correct mel data
    # from the file (encoder or decoder)
    if src_mel == "encoder":
        src_mel_idx = 0
    elif src_mel == "decoder":
        src_mel_idx = 1
    else:
        src_mel_idx = None

    mylogger.info("Processing mel data from : %s", src_mel)

    with torch.no_grad():
        with tqdm(
            range(len(filenames_list)),
            total=len(filenames_list),
            desc="Inference",
            position=1,
            dynamic_ncols=True,
        ) as progress_bar:
            for i in progress_bar:
                sample_id = filenames_list[i][:-4]  # remove .wav
                try:
                    # load generated mel data
                    mel = np.load(data_dir / f"{sample_id}.npy")
                    if len(mel.shape) == 2:
                        if mel.shape[0] == 161:  # (161, T) enc/dec/phnm_map
                            if src_mel == "encoder":
                                mel = mel[:80, :]
                            elif src_mel == "decoder":
                                mel = mel[80:160, :]
                            else:
                                mylogger.warning(
                                    "Mel data has 161 features, but src_mel is not specified. \
                                    Using first 80 features (encoder)."
                                )
                                mel = mel[:80, :]
                        elif mel.shape[0] == 80:  # (80, T) single sample
                            pass
                        else:
                            mylogger.error(
                                f"Unexpected shape for mel data: {mel.shape}. "
                                "Expected (161, T) for enc/dec/phnm_map. Or (80, T) for single sample"
                            )
                            continue

                    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)
                    audio = (
                        vocoder.forward(mel).cpu().squeeze().clamp(-1, 1).numpy()
                        * 32768
                    ).astype(np.int16)

                    save_path = save_dir / f"{sample_id}_{src_mel}.wav"
                    write(save_path, 22050, audio)

                    mylogger.info(f"Saved {save_path}")
                except FileNotFoundError as e:
                    mylogger.error(f"File not found for sample {sample_id}: {e}")

    mylogger.info("Inference completed. All files saved to %s", save_dir)
    mylogger.info("Total samples processed: %d", len(filenames_list))
