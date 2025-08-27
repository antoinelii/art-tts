import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
import importlib

import soundfile as sf
import torch

from voxcommunis.io import read_manifest
from model_ms.sparc_generator import SpkHiFiGANGenerator
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
    "--save_dir",  # ex: ../../data/VoxCommunis/test-20h/hifigan_pred/v6/grad_1000
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
    "--generator_ckpt",
    type=str,
    default="ckpt/sparc_multi.ckpt",  # or "ckpt/sparc_multi.ckpt"
)
parser.add_argument(
    "--version",
    type=str,  # v6
)
parser.add_argument(
    "--params_name",
    type=str,
    default="params_v6",
)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--src_art",
    type=str,
    default="",  # , choices=["encoder", "decoder", ""], "" for sparc
)

if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    manifest_path = Path(args.manifest_path)
    sparc_dir = Path(args.sparc_dir)
    device = args.device
    src_art = args.src_art
    version = args.version
    params_name = args.params_name
    generator_ckpt = Path(args.generator_ckpt)

    params = importlib.import_module(f"configs.{params_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    mylogger.info("Load model...")
    ckpt = torch.load(generator_ckpt)
    if "sparc" in generator_ckpt.name:
        wav_generator = SpkHiFiGANGenerator(
            spk_ft_size=1024, **ckpt["config"]["generator_configs"]
        )
        wav_generator.to(device)
        wav_generator.eval()
        wav_generator.spk_enc.load_state_dict(ckpt["state_dict"]["spk_ft"])
        wav_generator.generator.load_state_dict(ckpt["state_dict"]["generator"])
        wav_sr = ckpt["config"]["sr"]
        mylogger.info("Model loaded from %s", generator_ckpt)
    else:
        mylogger.error(
            f"Unsupported generator_ckpt: {generator_ckpt}. Supported: sparc."
        )
        exit(1)

    mylogger.info("Load filelist...")

    manifest = read_manifest(manifest_path)
    manifest_ids = set(manifest.keys())
    dir_files = list(data_dir.glob("*.npy"))
    dir_ids = set([f.stem for f in dir_files])
    file_ids = list(manifest_ids.intersection(dir_ids))

    mylogger.info("Found %d samples to process", len(file_ids))

    # if from arttts_pred need to extract the correct articulatory data
    # from the file (encoder or decoder)
    if src_art == "encoder":
        src_art_idx = 0
    elif src_art == "decoder":
        src_art_idx = 1
    else:
        src_art_idx = None

    mylogger.info("Processing articulatory data from : %s", src_art)

    with torch.no_grad():
        with tqdm(
            range(len(file_ids)),
            total=len(file_ids),
            desc="Inference",
            position=1,
            dynamic_ncols=True,
        ) as progress_bar:
            for i in progress_bar:
                sample_id = file_ids[i]
                try:
                    # load generated articulatory data
                    art = np.load(data_dir / f"{sample_id}.npy")
                    assert len(art.shape) == 2, (
                        f"Unexpected shape for articulatory data: {art.shape} != 2"
                    )
                    if art.shape[1] == 15:  # (T, 14) single sample
                        art = art[:, :14].T  # from sparc shape (T, 14) to (14, T)
                    elif art.shape[0] == 29:  # (29, T) enc/dec/phnm_map
                        if src_art == "encoder":
                            art = art[:14, :]
                        elif src_art == "decoder":
                            art = art[14:28, :]
                        else:
                            mylogger.warning(
                                "Articulatory data has 29 features, but src_art is not specified. \
                                Using second 14 features (decoder)."
                            )
                            art = art[14:28, :]
                    else:
                        mylogger.error(
                            f"Unexpected shape for articulatory data: {art.shape}. "
                            "Expected (T, 14) or (29, T) for enc/dec/phnm_map."
                        )
                        continue

                    sparc_ema = np.load(sparc_dir / f"emasrc/{sample_id}.npy")[
                        :, :14
                    ]  # (T, 14)
                    spk_ft = np.load(
                        sparc_dir / f"spk_preemb/{sample_id}.npy"
                    )  # (1024,)
                    if src_art:  # from arttts_pred
                        save_path = save_dir / f"{sample_id}_{src_art}.wav"
                        pitch_mu, pitch_std = (
                            sparc_ema[:, 12].mean(),
                            sparc_ema[:, 12].std(),
                        )
                        art[12, :] = art[12, :] * pitch_std + pitch_mu
                        if params.log_normalize_loudness:
                            loudness_mu, loudness_std = (
                                np.log(sparc_ema[:, 13] + 1e-9).mean(),
                                np.log(sparc_ema[:, 13] + 1e-9).std(),
                            )
                            art[13, :] = (
                                art[13, :] * loudness_std + loudness_mu
                            )  # denormalize log-loudness
                            art[13, :] = np.exp(art[13, :])  # delog log-loudness
                        art = torch.from_numpy(art).float().to(device)
                        art = art.unsqueeze(0)  # (1, 14, T)

                    else:  # from sparc generation
                        save_path = save_dir / f"{sample_id}.wav"
                        art = torch.from_numpy(sparc_ema).float().to(device)
                        art = art.unsqueeze(0).transpose(1, 2)  # (1, 14, T)

                    spk_ft = torch.from_numpy(spk_ft).float().to(device)
                    spk_ft = spk_ft.unsqueeze(0)  # (1, 1024)
                    with torch.no_grad():
                        wav = wav_generator(art, spk_ft)
                    wav = wav.squeeze().cpu().numpy()

                    sf.write(save_path, wav, wav_sr)
                    mylogger.info(f"Saved {save_path}")
                except FileNotFoundError as e:
                    mylogger.error(f"File not found for sample {sample_id}: {e}")

    mylogger.info("Inference completed. All files saved to %s", save_dir)
    mylogger.info("Total samples processed: %d", len(file_ids))
