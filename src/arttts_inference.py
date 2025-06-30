import numpy as np
import logging
from tqdm import tqdm
import importlib
from pathlib import Path

import torch

from model import GradTTS
from data_phnm import PhnmArticDataset, PhnmBatchCollate
from utils import TqdmLoggingHandler
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
    "--data_dir",  # ex: ../../data
    type=str,
)
parser.add_argument(
    "--save_dir",  # ex: ../../data/LJSpeech-1.1/arttts_pred/v1/grad_565
    type=str,
)
parser.add_argument(
    "--filelist_path",  # ex resources/filelists/ljspeech/valid_v1.txt
    type=str,
)
parser.add_argument("--version", type=str, default="v1")
parser.add_argument("--ckpt_name", type=str, default="grad_565.pt")
parser.add_argument("--params_name", type=str, default="params_v1")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size for inference, default is 16"
)


if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    filelist_path = Path(args.filelist_path)
    version = args.version
    ckpt_name = args.ckpt_name
    params_name = args.params_name
    device = args.device
    batch_size = args.batch_size

    save_dir.mkdir(parents=True, exist_ok=True)
    params = importlib.import_module(f"configs.{params_name}")

    mylogger.info("Start model init...")

    model = GradTTS(
        params.n_ipa_feats,
        params.n_spks,
        None if params.n_spks == 1 else params.spk_emb_dim,  # spk_emb_dim
        params.n_enc_channels,
        params.filter_channels,
        params.filter_channels_dp,
        params.n_heads,
        params.n_enc_layers,
        params.enc_kernel,
        params.enc_dropout,
        params.window_size,
        params.n_feats,
        params.dec_dim,
        params.beta_min,
        params.beta_max,
        params.pe_scale,
    ).to(device)

    ckpt_filepath = CKPT_DIR / version / ckpt_name
    ckpt_state_dict = torch.load(ckpt_filepath, map_location=torch.device(device))

    model.load_state_dict(ckpt_state_dict)

    mylogger.info("Model loaded from %s", ckpt_filepath)

    mylogger.info("Start dataset init...")

    dataset = PhnmArticDataset(
        filelist_path,
        data_root_dir=data_dir,
        load_coder=False,
        merge_diphtongues=params.merge_diphtongues,
    )

    mylogger.info("Dataset loaded with %d samples", len(dataset))

    reorder_feats = params.reorder_feats
    filepaths_list = dataset.filepaths_list
    collator = PhnmBatchCollate()

    good_sample = dataset.get_phnm_emb(
        "DUMMY/MNGU0/arttts/s1/phnm3/mngu0_s1_0001_phnm3.npy"
    )
    len_good = good_sample.shape[-1]
    # around 3 seconds (39 phnms) good sample, perfect to reach sufficient length

    model.eval()
    with torch.no_grad():
        with tqdm(
            range(0, len(filepaths_list), batch_size),
            total=len(filepaths_list) // batch_size,
            desc="Inference",
            position=1,
            dynamic_ncols=True,
        ) as progress_bar:
            for i in progress_bar:
                batch_filepaths = filepaths_list[i : i + batch_size]
                phnm3_filepaths = [fp[1] for fp in batch_filepaths]
                phnm_embs = [
                    {
                        "x": torch.cat(
                            (good_sample, dataset.get_phnm_emb(phnm3_fp)), dim=1
                        )
                    }
                    for phnm3_fp in phnm3_filepaths
                ]
                batch = collator(phnm_embs)
                x = batch["x"].to(torch.float32)
                x_lengths = batch["x_lengths"]
                y_enc, y_dec, attn = model(
                    x, x_lengths, n_timesteps=50
                )  # (B, 16, T) x 2 , (B,1,T0,T)
                y_enc_14 = y_enc[:, reorder_feats, :].detach().cpu()
                y_dec_14 = y_dec[:, reorder_feats, :].detach().cpu()
                for j, (filepath, y_enc_j, y_dec_j) in enumerate(
                    zip(batch_filepaths, y_enc_14, y_dec_14)
                ):
                    x_len = x_lengths[j]
                    attn_j = attn[j, 0, :x_len, :].detach().cpu()  # (x_len, y_len_max)
                    y_len_good = np.max(np.where(attn_j[len_good - 1]))
                    y_len_tot = np.max(
                        np.where(attn_j[-1])
                    )  # last row, last value=1 index (binary attn)
                    sample_id = filepath[0].split("/")[-1][:-4]
                    save_path = save_dir / f"{sample_id}.npy"
                    y_enc_dec_j = np.array(
                        [
                            y_enc_j[:, y_len_good + 1 : y_len_tot + 1].numpy(),
                            y_dec_j[:, y_len_good + 1 : y_len_tot + 1].numpy(),
                        ]
                    )  # (2, 14, T)
                    np.save(save_path, y_enc_dec_j)
                    mylogger.info(f"Saved {save_path}")

    mylogger.info("Inference completed. All files saved to %s", save_dir)
    mylogger.info("Total samples processed: %d", len(filepaths_list))
