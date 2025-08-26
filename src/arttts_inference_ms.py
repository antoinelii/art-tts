import numpy as np
import logging
from tqdm import tqdm
import importlib
from pathlib import Path

import torch

from model_ms import GradTTArtic
from data_ms import PhnmDataset, PhnmBatchCollate

from voxcommunis.decoder import FeatureDecoder
from voxcommunis.data import FeatureTokenizer

from utils import TqdmLoggingHandler
from paths import CKPT_DIR

import argparse

mylogger = logging.getLogger(__name__)
mylogger.setLevel(logging.INFO)

handler = TqdmLoggingHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
mylogger.addHandler(handler)


def init_model(params, device):
    model = GradTTArtic(
        n_ipa_feats=params.n_ipa_feats,  # 26, 24 phonological traits + 1 silence dim + 1 phoneme repetition count
        spk_emb_dim=params.spk_emb_dim,  # 64
        n_enc_channels=params.n_enc_channels,  # 192
        filter_channels=params.filter_channels,  # 768
        filter_channels_dp=params.filter_channels_dp,  # 256
        n_heads=params.n_heads,  # 2
        n_enc_layers=params.n_enc_layers,  # 6
        enc_kernel=params.enc_kernel,  # 3
        enc_dropout=params.enc_dropout,  # 0.1
        window_size=params.window_size,  # 4
        n_feats=params.n_feats,  # 16 (articulatory features)
        dec_dim=params.dec_dim,  # 64
        beta_min=params.beta_min,  # 0.05
        beta_max=params.beta_max,  # 20.0
        pe_scale=params.pe_scale,  # 1000
        spk_preemb_dim=1024,  # Similar
    ).to(device)
    return model


def get_dataset(dataset_dir, manifest_path, alignment_path, separate_files, tokenizer):
    dataset = PhnmDataset(
        dataset_dir=dataset_dir,
        separate_files=separate_files,
        manifest_path=manifest_path,
        alignment_path=alignment_path,
        feature_tokenizer=tokenizer,
        reorder_feats=params.reorder_feats,
        pitch_idx=params.pitch_idx,
        loudness_idx=params.loudness_idx,
        log_normalize_loudness=params.log_normalize_loudness,
        random_seed=params.random_seed,
    )
    return dataset


def get_inputs(batch_file_ids, dataset, collator):
    batch = []
    for file_id in batch_file_ids:
        phon_features = dataset.get_phon_feats(file_id)  # (n_ipa_feats, seq_len)
        spk_preemb = dataset.get_spk_features(file_id)  # (1024,)
        batch.append(
            {
                "x": phon_features,
                "spk_ft": spk_preemb,
            }
        )
    batch = collator(batch)  # in place
    x, x_lengths, spk_ft = batch["x"], batch["x_lengths"], batch["spk_ft"]
    return x, x_lengths, spk_ft


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
)
parser.add_argument(
    "--save_dir",
    type=str,
)
parser.add_argument(
    "--manifest_path",
    type=str,
)
parser.add_argument(
    "--alignment_path",
    type=str,
)
parser.add_argument("--version", type=str, default="v6")
parser.add_argument("--ckpt_name", type=str, default="grad_1000.pt")
parser.add_argument("--params_name", type=str, default="params_v6")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size for inference, default is 1 and should be done with 1\
        sample at a time to avoid messing up with group normalization",  # should remove groupnorm in future versions
)

parser.add_argument(
    "--max_samples",
    type=int,
    default=0,
    help="Maximum number of samples to process. 0 means all samples (default: 0)",
)

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    save_dir = Path(args.save_dir)
    manifest_path = Path(args.manifest_path)
    alignment_path = Path(args.alignment_path)
    if manifest_path.is_file():
        separate_files = False
    else:  # directory with multiple manifests
        separate_files = True
    version = args.version
    ckpt_name = args.ckpt_name
    params_name = args.params_name
    device = args.device
    batch_size = args.batch_size

    save_dir.mkdir(parents=True, exist_ok=True)
    params = importlib.import_module(f"configs.{params_name}")

    mylogger.info("Start model init...")

    model = init_model(params, device)

    ckpt_filepath = CKPT_DIR / version / ckpt_name
    ckpt_state_dict = torch.load(ckpt_filepath, map_location=torch.device(device))

    model.load_state_dict(ckpt_state_dict)

    mylogger.info("Model loaded from %s", ckpt_filepath)

    mylogger.info("Getting dataset ...")

    fd = FeatureDecoder(sum_diphthong=True)
    tokenizer = FeatureTokenizer(fd)
    dataset = get_dataset(
        dataset_dir,
        manifest_path,
        alignment_path,
        separate_files=separate_files,
        tokenizer=tokenizer,
    )

    mylogger.info("Dataset loaded with %d samples", len(dataset))

    reorder_feats = params.reorder_feats

    file_ids = [e[0] for e in dataset.manifest]
    if args.max_samples > 0:
        file_ids = file_ids[: args.max_samples]
        mylogger.info("Process only the first %d samples", args.max_samples)

    collator = PhnmBatchCollate()

    model.eval()
    with torch.no_grad():
        with tqdm(
            range(0, len(file_ids), batch_size),
            total=len(file_ids) // batch_size,
            desc="Inference",
            position=1,
            dynamic_ncols=True,
        ) as progress_bar:
            for i in progress_bar:
                # Prepare batch
                batch_file_ids = file_ids[i : i + batch_size]
                # Run inference
                x, x_lengths, spk_ft = get_inputs(batch_file_ids, dataset, collator)
                x = x.to(device)
                x_lengths = x_lengths.to(device)
                spk_ft = spk_ft.to(device)
                y_enc, y_dec, attn = model(
                    x, x_lengths, spk_ft, n_timesteps=50
                )  # (B, 16, T) x 2 , (B,1,T0,T)

                y_enc_ = y_enc[:, reorder_feats, :].detach().cpu()  # (B, 14, T)
                y_dec_ = y_dec[:, reorder_feats, :].detach().cpu()  # (B, 14, T)

                for j, (file_id, y_enc_j, y_dec_j) in enumerate(
                    zip(batch_file_ids, y_enc_, y_dec_)
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
                    save_path = save_dir / f"{file_id}.npy"
                    y_enc_dec_j = np.vstack(
                        [
                            y_enc_j[:, : y_len + 1].numpy(),
                            y_dec_j[:, : y_len + 1].numpy(),
                            input_map,
                        ]
                    )  # (29, T)  # 29 (14 enc + 14 dec + 1 input_map)
                    np.save(save_path, y_enc_dec_j)
                    mylogger.info(f"Saved {save_path}")

    mylogger.info("Inference completed. All files saved to %s", save_dir)
    mylogger.info("Total samples processed: %d", len(file_ids))
