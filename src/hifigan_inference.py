import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path

import soundfile as sf
import torch

from sparc import load_model
from utils import TqdmLoggingHandler, parse_filelist
from paths import CKPT_DIR

import argparse

mylogger = logging.getLogger(__name__)
mylogger.setLevel(logging.INFO)

handler = TqdmLoggingHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
mylogger.addHandler(handler)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",  # ex: ../../data/LJSpeech-1.1/arttts_pred/v1/grad_200
    type=str,
)
parser.add_argument(
    "--save_dir",  # ex: ../../data/LJSpeech-1.1/hifigan_pred/v1/grad_200
    type=str,
)
parser.add_argument(
    "--filelist_path",  # ex resources/filelists/ljspeech/valid_v1.txt
    type=str,
)
parser.add_argument(
    "--main_dir",  # ex: ../../data
    type=str,
)
parser.add_argument(
    "--sparc_ckpt",
    type=str,
    default="ckpt/sparc_en.ckpt",  # or "ckpt/sparc_multi.ckpt"
)
parser.add_argument(
    "--version",
    type=str,  # bool
)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--src_art",
    type=str,
    default="",  # , choices=["encoder", "decoder"],
)
parser.add_argument(
    "--mean_spk_emb",
    type=int,  # bool
    default=0,  # whether to use mean speaker embedding (0 or 1)
)

if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    filelist_path = Path(args.filelist_path)
    main_dir = Path(args.main_dir)
    device = args.device
    src_art = args.src_art
    version = args.version
    mean_spk_emb = args.mean_spk_emb

    if version == "v1_1":
        unnorm_loudness = True  # whether to use unnormalized loudness
    else:
        unnorm_loudness = False

    save_dir.mkdir(parents=True, exist_ok=True)

    mylogger.info("Load model...")

    ckpt_filepath = CKPT_DIR / "sparc_en.ckpt"
    coder = load_model(ckpt=ckpt_filepath, device=device)
    wav_sr = coder.output_sr

    mylogger.info("Model loaded from %s", ckpt_filepath)

    mylogger.info("Load filelist...")

    filepaths_list = parse_filelist(filelist_path)
    filenames_list = [fp[0].split("/")[-1] for fp in filepaths_list]

    mylogger.info("Filelist loaded with %d samples", len(filenames_list))

    # if from arttts_pred need to extract the correct articulatory data
    # from the file (encoder or decoder)
    if src_art == "encoder":
        src_art_idx = 0
    elif src_art == "decoder":
        src_art_idx = 1
    else:
        src_art_idx = None

    mylogger.info("Processing articulatory data from : %s", src_art)

    if version in ["v1", "v1_1"]:
        phnm3_filepath = filepaths_list[0][1].replace("DUMMY/", str(main_dir) + "/")
        encoded_audio_dir = Path(phnm3_filepath).parents[1] / "encoded_audio_en"
    elif version == "v4":
        # for v4 we use the same directory as the data_dir
        emasrc_filepath = filepaths_list[0][0].replace("DUMMY/", str(main_dir) + "/")
        encoded_audio_dir = Path(emasrc_filepath).parents[1]
    else:
        mylogger.error(
            f"Unsupported version: {version}. Supported versions are: v1, v1_1, v4."
        )
        exit(1)

    mylogger.info("Encoded audio directory: %s", encoded_audio_dir)

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
                    # load generated articulatory data
                    art = np.load(data_dir / f"{sample_id}.npy")
                    if len(art.shape) == 2:
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
                                    Using first 14 features (encoder)."
                                )
                                art = art[:14, :]
                        else:
                            mylogger.error(
                                f"Unexpected shape for articulatory data: {art.shape}. "
                                "Expected (T, 14) or (29, T) for enc/dec/phnm_map."
                            )
                            continue

                    # old arttts_inference method handling
                    elif len(art.shape) == 3:  # (2, 14, T)  enc/dec
                        if src_art_idx is None:
                            mylogger.warning(
                                "Articulatory data has 3 dimensions, but src_art is not specified. \
                                Using first index (encoder)."
                            )
                            src_art_idx = 0
                        art = art[src_art_idx, :14, :]

                    # load preliminary data
                    sparc_ema = np.load(
                        encoded_audio_dir / "emasrc" / f"{sample_id}.npy"
                    )

                    if mean_spk_emb:
                        # Use mean speaker embedding if specified
                        spk_emb = np.load(
                            encoded_audio_dir / "spk_emb" / "mean_spk_emb.npy"
                        )
                    else:
                        spk_emb = np.load(
                            encoded_audio_dir / "spk_emb" / f"{sample_id}.npy"
                        )
                    if src_art:  # from arttts_pred
                        if mean_spk_emb:
                            save_path = (
                                save_dir / f"{sample_id}_{src_art}_mean_spk_emb.wav"
                            )
                        else:
                            save_path = save_dir / f"{sample_id}_{src_art}.wav"
                        pitch_mu, pitch_std = (
                            sparc_ema[:, 12].mean(),
                            sparc_ema[:, 12].std(),
                        )
                        if unnorm_loudness:
                            loudness_mu, loudness_std = (
                                np.log(sparc_ema[:, 13] + 1e-9).mean(),
                                np.log(sparc_ema[:, 13] + 1e-9).std(),
                            )
                            art[13, :] = (
                                art[13, :] * loudness_std + loudness_mu
                            )  # denormalize log-loudness
                            art[13, :] = np.exp(art[13, :])  # delog log-loudness
                        code_art = {
                            "ema": art[:12, :].T,  # (T, 12)
                            "loudness": art[13, :],
                            "pitch": (
                                art[12, :] * pitch_std + pitch_mu
                            ),  # denormalize pitch
                            "spk_emb": spk_emb,
                        }

                    else:  # from sparc generation
                        save_path = save_dir / f"{sample_id}.wav"
                        code_art = {
                            "ema": art[:12, :].T,  # (T, 12)
                            "loudness": art[13, :],
                            "pitch": art[12, :],  # already good scaled
                            "spk_emb": spk_emb,
                        }
                    wav = coder.decode(**code_art)

                    sf.write(save_path, wav, wav_sr)
                    mylogger.info(f"Saved {save_path}")
                except FileNotFoundError as e:
                    mylogger.error(f"File not found for sample {sample_id}: {e}")

    mylogger.info("Inference completed. All files saved to %s", save_dir)
    mylogger.info("Total samples processed: %d", len(filenames_list))


# Maybe useful later
# def batch_collator(arts, spk_embs, pitch_means, pitch_stds):
#        """
#        Collate function for sparc hifigan decoder inputs.
#        Args:
#            arts (list): List of articulatory features tensors of shape (n_feats, T)
#            spk_embs (list): List of speaker embeddings of shape (n_spk_emb,).
#            pitch_means (list): List of pitch means.
#            pitch_stds (list): List of pitch standard deviations.
#        Returns:
#        """
#        B = len(arts)
#        art_max_length = max([art.shape[-1] for art in arts])
#        n_feats = arts[0].shape[-2]
#
#        art_batch = np.zeros((B, n_feats, art_max_length), dtype=np.float32)
#        x_lengths = []
#
#        for i, art in enumerate(arts):
#            x_lengths.append(art.shape[-1])
#            art_batch[i, :, : art.shape[-1]] = art
#
#        ema = art_batch[:, :12, :].transpose(0,2,1)  # (B, T, 12)
#        loudness = art_batch[:, 13, :]  # (B, T)
#        pitch = art_batch[:, 12, :]  # (B, T)
#        pitch *= np.array(pitch_stds).reshape(B, 1)
#        pitch += np.array(pitch_means).reshape(B, 1)
#        return {
#            "ema": ema,
#            "loudness": loudness[:,:,None],
#            "pitch": pitch[:,:,None],
#            "spk_emb": np.array(spk_embs),}
