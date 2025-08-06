import numpy as np
import logging
from tqdm import tqdm
import importlib
from pathlib import Path

import torch

from model import ArtTTS, GradTTS, AttentionTTS
from data_phnm import PhnmArticDataset, PhnmBatchCollate
from data_textmel import TextMelDataset, TextBatchCollate
from data_textart import TextArtDataset
from data_phnmmel import PhnmMelDataset
from text.symbols import symbols
from utils import TqdmLoggingHandler
from paths import CKPT_DIR

import argparse

mylogger = logging.getLogger(__name__)
mylogger.setLevel(logging.INFO)

handler = TqdmLoggingHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
mylogger.addHandler(handler)

phnm_versions = ["v1", "v1_", "v1_1", "v3"]
text_versions = ["v2", "v4"]
attention_versions = ["v5", "v5_preblock"]
artic_versions = ["v1", "v1_", "v1_1", "v4", "v5", "v5_preblock"]
mel_versions = ["v2", "v3"]


def init_model(version, params, device):
    if version in phnm_versions:
        model = ArtTTS(
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
    elif version in text_versions:
        add_blank = params.add_blank
        nsymbols = len(symbols) + 1 if add_blank else len(symbols)
        model = GradTTS(
            nsymbols,
            params.n_spks,
            None if params.n_spks == 1 else params.spk_embed_dim,
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
    elif version in attention_versions:
        model = AttentionTTS(
            params.n_ipa_feats,
            params.n_spks,
            None if params.n_spks == 1 else params.spk_embed_dim,
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

    else:
        raise ValueError(f"Unsupported version: {version}")
    return model


def init_dataset(version, params, data_dir, filelist_path):
    if version in ["v1", "v1_"]:
        dataset = PhnmArticDataset(
            filelist_path,
            data_root_dir=data_dir,
            load_coder=False,
            merge_diphtongues=params.merge_diphtongues,
        )
    elif version in ["v1_1", "v5", "v5_preblock"]:
        dataset = PhnmArticDataset(
            filelist_path,
            data_root_dir=data_dir,
            load_coder=True,
            merge_diphtongues=params.merge_diphtongues,
            # not necessary for inference to normalize loudness
            log_normalize_loudness=params.log_normalize_loudness,
        )
    elif version in ["v2"]:
        dataset = TextMelDataset(
            filelist_path,
            data_root_dir=data_dir,
            cmudict_path=params.cmudict_path,
            add_blank=params.add_blank,
            gradtts_text_conv=params.gradtts_text_conv,
        )
    elif version in ["v3"]:
        dataset = PhnmMelDataset(
            filelist_path,
            data_root_dir=data_dir,
        )
    elif version in ["v4"]:
        dataset = TextArtDataset(
            filelist_path,
            data_root_dir=data_dir,
            cmudict_path=params.cmudict_path,
            add_blank=params.add_blank,
            gradtts_text_conv=params.gradtts_text_conv,
        )
    else:
        raise ValueError(f"Unsupported version: {version}")
    return dataset


def get_collator(version, params):
    if version in phnm_versions:
        collator = PhnmBatchCollate()
    elif version in text_versions:
        collator = TextBatchCollate()
    else:
        raise ValueError(f"Unsupported version: {version}")
    return collator


def get_aligned_inputs(batch_filepaths, dataset, collator):
    phnm3_filepaths = [fp[1] for fp in batch_filepaths]
    phnm_embs = [{"x": dataset.get_phnm_emb(phnm3_fp)} for phnm3_fp in phnm3_filepaths]
    batch = collator(phnm_embs)
    x = batch["x"].to(torch.float32)
    x_lengths = batch["x_lengths"]

    x_durs = [
        dataset.get_x_durations(phnm3_fp, merge_diphtongues=dataset.merge_diphtongues)
        for phnm3_fp in phnm3_filepaths
    ]
    x_durations = torch.zeros((len(x_durs), x.shape[-1]), dtype=torch.float32)
    for i, durs in enumerate(x_durs):
        x_durations[i, : len(durs)] = durs
    return x, x_lengths, x_durations


def get_phnm_inputs(batch_filepaths, dataset, collator):
    phnm3_filepaths = [fp[1] for fp in batch_filepaths]
    phnm_embs = [{"x": dataset.get_phnm_emb(phnm3_fp)} for phnm3_fp in phnm3_filepaths]
    batch = collator(phnm_embs)
    x = batch["x"].to(torch.float32)
    x_lengths = batch["x_lengths"]
    return x, x_lengths


def get_text_inputs(batch_filepaths, dataset, collator):
    text_filepaths = [fp[1] for fp in batch_filepaths]
    text_embs = [{"x": dataset.get_text(text_fp)} for text_fp in text_filepaths]
    batch = collator(text_embs)
    x = batch["x"].to(torch.long)
    x_lengths = batch["x_lengths"]
    return x, x_lengths


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
    "--batch_size",
    type=int,
    default=1,
    help="Batch size for inference, default is 1 and should be done with 1\
        sample at a time to avoid messing up with group normalization",  # should remove groupnorm in future versions
)
parser.add_argument(
    "--use_align",
    type=int,  # bool
    default=0,
    help="Whether to use alignment for inference. If True, will create x_durations from dataset phoneme alignment",
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
    use_align = int(args.use_align)  # bool

    save_dir.mkdir(parents=True, exist_ok=True)
    params = importlib.import_module(f"configs.{params_name}")

    mylogger.info("Start model init...")

    model = init_model(version, params, device)

    ckpt_filepath = CKPT_DIR / version / ckpt_name
    ckpt_state_dict = torch.load(ckpt_filepath, map_location=torch.device(device))

    model.load_state_dict(ckpt_state_dict)

    mylogger.info("Model loaded from %s", ckpt_filepath)

    mylogger.info("Start dataset init...")

    dataset = init_dataset(
        version,
        params,
        data_dir,
        filelist_path,
    )

    mylogger.info("Dataset loaded with %d samples", len(dataset))

    if version in artic_versions:
        reorder_feats = params.reorder_feats
    filepaths_list = dataset.filepaths_list
    collator = get_collator(version, params)

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
                # Prepare batch
                batch_filepaths = filepaths_list[i : i + batch_size]

                # Run inference
                if use_align:
                    x, x_lengths, x_durations = get_aligned_inputs(
                        batch_filepaths,
                        dataset,
                        collator,
                    )
                    # Run inference
                    y_enc, y_dec, attn = model(
                        x,
                        x_lengths,
                        n_timesteps=50,
                        x_durations=x_durations,
                    )  # (B, 16, T) x 2 , (B,1,T0,T)
                elif (version in phnm_versions) or (version in attention_versions):
                    x, x_lengths = get_phnm_inputs(batch_filepaths, dataset, collator)
                    y_enc, y_dec, attn = model(
                        x, x_lengths, n_timesteps=50
                    )  # (B, 16, T) x 2 , (B,1,T0,T)
                elif version in text_versions:
                    x, x_lengths = get_text_inputs(batch_filepaths, dataset, collator)
                    y_enc, y_dec, attn = model(
                        x, x_lengths, n_timesteps=50
                    )  # (B, 16, T) x 2 , (B,1,T0

                if version in artic_versions:
                    y_enc_ = y_enc[:, reorder_feats, :].detach().cpu()  # (B, 14, T)
                    y_dec_ = y_dec[:, reorder_feats, :].detach().cpu()  # (B, 14, T)
                else:
                    y_enc_ = y_enc.detach().cpu()  # (B, 80, T)
                    y_dec_ = y_dec.detach().cpu()  # (B, 80, T)
                for j, (filepath, y_enc_j, y_dec_j) in enumerate(
                    zip(batch_filepaths, y_enc_, y_dec_)
                ):
                    x_len = x_lengths[j]
                    attn_j = attn[j, 0, :x_len, :].detach().cpu()  # (x_len, y_len_max)
                    input_map = np.where(
                        attn_j == 1
                    )[
                        0
                    ]  # (y_len,)  easier storage with enc & dec gives the index of the matching input
                    y_len = np.max(
                        np.where(attn_j[-1])
                    )  # last row, last value=1 index (binary attn)
                    sample_id = filepath[0].split("/")[-1][:-4]
                    save_path = save_dir / f"{sample_id}.npy"
                    y_enc_dec_j = np.vstack(
                        [
                            y_enc_j[:, : y_len + 1].numpy(),
                            y_dec_j[:, : y_len + 1].numpy(),
                            input_map,
                        ]
                    )  # (29, T)  # 29 (14 enc + 14 dec + 1 input_map)
                    # or (161, T) for mel_versions (80 enc + 80 dec + 1 input_map)
                    np.save(save_path, y_enc_dec_j)
                    mylogger.info(f"Saved {save_path}")

    mylogger.info("Inference completed. All files saved to %s", save_dir)
    mylogger.info("Total samples processed: %d", len(filepaths_list))
