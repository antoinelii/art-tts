# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from configs import params_v6
from model_ms import GradTTArtic
from data_ms import PhnmArticDataset, PhnmArticBatchCollate
from voxcommunis.sampler import LengthGroupedSampler, LengthGroupedLanguageUpSampler
from voxcommunis.decoder import FeatureDecoder
from voxcommunis.data import FeatureTokenizer
from utils import (
    plot_tensor,
    save_plot,
    plot_art_14,
    save_plot_art_14,
    TqdmLoggingHandler,
)
from metrics import normalized_dtw_score

log_dir = Path(params_v6.log_dir)
reorder_feats = params_v6.reorder_feats

# Setup logger
mylogger = logging.getLogger(__name__)
mylogger.setLevel(logging.INFO)

handler = TqdmLoggingHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
mylogger.addHandler(handler)

start_epoch = 1
end_epoch = 5000
val_every = params_v6.val_every
save_every = params_v6.save_every


def get_train_dataset(tokenizer):
    train_dataset = PhnmArticDataset(
        dataset_dir=params_v6.dataset_dir,
        separate_files=params_v6.separate_files,
        manifest_path=params_v6.train_manifest,
        alignment_path=params_v6.train_alignment,
        feature_tokenizer=tokenizer,
        reorder_feats=params_v6.reorder_feats,
        pitch_idx=params_v6.pitch_idx,
        loudness_idx=params_v6.loudness_idx,
        log_normalize_loudness=params_v6.log_normalize_loudness,
        random_seed=params_v6.random_seed,
    )
    return train_dataset


def get_valid_dataset(tokenizer):
    valid_dataset = PhnmArticDataset(
        dataset_dir=params_v6.dataset_dir,
        separate_files=params_v6.separate_files,
        manifest_path=params_v6.val_manifest,
        alignment_path=params_v6.val_alignment,
        feature_tokenizer=tokenizer,
        reorder_feats=params_v6.reorder_feats,
        pitch_idx=params_v6.pitch_idx,
        loudness_idx=params_v6.loudness_idx,
        log_normalize_loudness=params_v6.log_normalize_loudness,
        random_seed=params_v6.random_seed,
    )
    return valid_dataset


def init_data_loader(
    dataset,
    validation: bool,
    multilingual: bool = False,
):
    batch_collate = PhnmArticBatchCollate()
    if validation:
        loader = DataLoader(
            dataset=dataset,
            batch_size=params_v6.batch_size,
            collate_fn=batch_collate,
            drop_last=False,
            num_workers=3,
            shuffle=False,
            pin_memory=True,
        )
        return loader
    else:  # training
        lengths = [dataset.manifest[i][1][1] for i in range(len(dataset))]
        if multilingual:
            sampler = LengthGroupedLanguageUpSampler(
                batch_size=params_v6.batch_size,
                lengths=lengths,
                lang_sizes=dataset.lang_sizes,
                generator=torch.manual_seed(params_v6.random_seed),
                upsample_factor=0.5,
            )
        else:
            sampler = LengthGroupedSampler(
                batch_size=params_v6.batch_size,
                lengths=lengths,
                generator=torch.manual_seed(params_v6.random_seed),
            )
        loader = DataLoader(
            dataset,
            batch_size=params_v6.batch_size,
            collate_fn=batch_collate,
            drop_last=True,
            sampler=sampler,
            num_workers=3,
            shuffle=False,
        )
    return loader


def log_valid_batch(logger, valid_dataset):
    """done only on rank 0"""
    valid_batch = valid_dataset.sample_test_batch(size=params_v6.test_size)
    for i, item in enumerate(valid_batch):
        art = item["y"][reorder_feats, :].cpu()
        logger.add_image(
            f"image_{i}/ground_truth",
            plot_art_14([art])[1],
            global_step=0,
            dataformats="HWC",
        )
        save_plot_art_14([art], f"{params_v6.log_dir}/original_{i}.png")
    return valid_batch


def init_model():
    model = GradTTArtic(
        n_ipa_feats=params_v6.n_ipa_feats,  # 26, 24 phonological traits + 1 silence dim + 1 phoneme repetition count
        spk_emb_dim=params_v6.spk_emb_dim,  # 64
        n_enc_channels=params_v6.n_enc_channels,  # 192
        filter_channels=params_v6.filter_channels,  # 768
        filter_channels_dp=params_v6.filter_channels_dp,  # 256
        n_heads=params_v6.n_heads,  # 2
        n_enc_layers=params_v6.n_enc_layers,  # 6
        enc_kernel=params_v6.enc_kernel,  # 3
        enc_dropout=params_v6.enc_dropout,  # 0.1
        window_size=params_v6.window_size,  # 4
        n_feats=params_v6.n_feats,  # 16 (articulatory features)
        dec_dim=params_v6.dec_dim,  # 64
        beta_min=params_v6.beta_min,  # 0.05
        beta_max=params_v6.beta_max,  # 20.0
        pe_scale=params_v6.pe_scale,  # 1000
        spk_preemb_dim=1024,  # Similar
    ).cuda()
    return model


def train_loop(model, train_loader, optimizer, epoch):
    model.train()
    prior_losses = []
    diff_losses = []
    losses = []
    enc_grad_norms = []
    dec_grad_norms = []

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        x, x_lengths = batch["x"].cuda(), batch["x_lengths"].cuda()
        y, y_lengths = batch["y"].cuda(), batch["y_lengths"].cuda()
        spk_ft = batch["spk_ft"].cuda()
        prior_loss, diff_loss = model.compute_loss(
            x, x_lengths, y, y_lengths, spk_ft, out_size=params_v6.out_size
        )
        loss = sum([prior_loss, diff_loss])
        loss.backward()

        enc_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.encoder.parameters(), max_norm=1
        )
        dec_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.decoder.parameters(), max_norm=1
        )
        optimizer.step()

        prior_losses.append(prior_loss.item())
        diff_losses.append(diff_loss.item())
        losses.append(loss.item())
        enc_grad_norms.append(enc_grad_norm.cpu().numpy())
        dec_grad_norms.append(dec_grad_norm.cpu().numpy())

    mean_train_prior_loss = np.mean(prior_losses)
    mean_train_diff_loss = np.mean(diff_losses)
    mean_train_loss = np.mean(losses)
    max_enc_grad_norm = np.max(enc_grad_norms)
    max_dec_grad_norm = np.max(dec_grad_norms)

    means_tensor = torch.tensor(
        [
            mean_train_prior_loss,
            mean_train_diff_loss,
            mean_train_loss,
        ],
    )
    maxs_tensor = torch.tensor(
        [max_enc_grad_norm, max_dec_grad_norm],
    )
    return (
        means_tensor,
        maxs_tensor,
    )


def save_train_scalars(logger, epoch, avg_losses, max_grad_norms):
    """Save training scalars to TensorBoard.
    Args:
        logger: TensorBoard logger
        epoch: Current epoch number
        avg_losses: Tensor containing mean loss values
        max_grad_norms: Tensor containing maximum gradient norms
    """

    mean_train_prior_loss = avg_losses[0].detach().cpu().numpy()
    mean_train_diff_loss = avg_losses[1].detach().cpu().numpy()
    mean_train_loss = avg_losses[2].detach().cpu().numpy()

    max_enc_grad_norm = max_grad_norms[0].detach().cpu().numpy()
    max_dec_grad_norm = max_grad_norms[1].detach().cpu().numpy()

    logger.add_scalar("training/prior_loss", mean_train_prior_loss, global_step=epoch)
    logger.add_scalar(
        "training/diffusion_loss", mean_train_diff_loss, global_step=epoch
    )
    logger.add_scalar("training/loss", mean_train_loss, global_step=epoch)
    logger.add_scalar(
        "training/encoder_grad_norm", max_enc_grad_norm, global_step=epoch
    )
    logger.add_scalar(
        "training/decoder_grad_norm", max_dec_grad_norm, global_step=epoch
    )

    log_msg = "Epoch %d:" % epoch
    log_msg += " prior loss = %.3f " % mean_train_prior_loss
    log_msg += "| diffusion loss = %.3f\n" % mean_train_diff_loss
    log_msg += "| loss = %.3f\n" % mean_train_loss
    with open(f"{log_dir}/train.log", "a") as f:
        f.write(log_msg)
    return log_msg


def validation_loop(model, val_loader):
    val_prior_losses = []
    val_diff_losses = []
    val_losses = []
    for batch_idx, batch in enumerate(val_loader):
        with torch.no_grad():
            x, x_lengths = batch["x"].cuda(), batch["x_lengths"].cuda()
            y, y_lengths = batch["y"].cuda(), batch["y_lengths"].cuda()
            spk_ft = batch["spk_ft"].cuda()
            prior_loss, diff_loss = model.compute_loss(
                x, x_lengths, y, y_lengths, spk_ft, out_size=params_v6.out_size
            )
            val_loss = sum([prior_loss, diff_loss])

            val_prior_losses.append(prior_loss.item())
            val_diff_losses.append(diff_loss.item())
            val_losses.append(val_loss.item())
    mean_val_prior_loss = np.mean(val_prior_losses)
    mean_val_diff_loss = np.mean(val_diff_losses)
    mean_val_loss = np.mean(val_losses)
    return [
        mean_val_prior_loss,
        mean_val_diff_loss,
        mean_val_loss,
    ]


def save_val_scalars(logger, epoch, val_losses):
    """Save validation scalars to TensorBoard.
    Args:
        logger: TensorBoard logger
        epoch: Current epoch number
        val_losses: List of validation loss values
    """
    mean_val_prior_loss = val_losses[0]
    mean_val_diff_loss = val_losses[1]
    mean_val_loss = val_losses[2]

    logger.add_scalar(
        "validation/prior_loss",
        mean_val_prior_loss,
        global_step=epoch,
    )
    logger.add_scalar(
        "validation/diffusion_loss",
        mean_val_diff_loss,
        global_step=epoch,
    )
    logger.add_scalar("validation/loss", mean_val_loss, global_step=epoch)
    log_msg = "Epoch %d: " % epoch
    log_msg += " prior loss = %.3f " % mean_val_prior_loss
    log_msg += "| diffusion loss = %.3f\n" % mean_val_diff_loss
    log_msg += "| loss = %.3f\n" % mean_val_loss
    with open(f"{log_dir}/val.log", "a") as f:
        f.write(log_msg)
    return log_msg


def save_synthesis(logger, model, epoch, valid_batch):
    """Run inference on a list of filepaths and save the results.
    Args:
        model: Trained model for inference
    """
    gt_enc_dtw_scores = []
    gt_dec_dtw_scores = []
    with torch.no_grad():
        for i, item in enumerate(valid_batch):
            x = item["x"].to(torch.float32).unsqueeze(0).cuda()
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            spk_ft = item["spk_ft"].unsqueeze(0).cuda()
            y_enc, y_dec, attn = model(x, x_lengths, spk_ft, n_timesteps=50)

            y_enc_14 = y_enc[0, reorder_feats, :].T.cpu()  # (n_frames, n_feats)
            y_dec_14 = y_dec[0, reorder_feats, :].T.cpu()  # (n_frames, n_feats)
            y_gt = item["y"][reorder_feats, :].T.cpu().numpy()  # (n_frames, n_feats)

            # Compute DTW distance to targets
            dist_gt_enc, y_gt_enc_ada, y_enc_14_ada = normalized_dtw_score(
                y_gt, y_enc_14.numpy()
            )
            dist_gt_dec, y_gt_dec_ada, y_dec_14_ada = normalized_dtw_score(
                y_gt, y_dec_14.numpy()
            )
            gt_enc_dtw_scores.append(dist_gt_enc)
            gt_dec_dtw_scores.append(dist_gt_dec)

            logger.add_image(
                f"image_{i}/generated_enc",
                plot_art_14(
                    [y_enc_14.T],
                )[1][:, :, 1:],
                global_step=epoch,
                dataformats="HWC",
            )
            logger.add_image(
                f"image_{i}/generated_dec",
                plot_art_14(
                    [y_dec_14.T],
                )[1][:, :, 1:],
                global_step=epoch,
                dataformats="HWC",
            )
            logger.add_image(
                f"image_{i}/alignment",
                plot_tensor(attn.squeeze().cpu())[:, :, 1:],
                global_step=epoch,
                dataformats="HWC",
            )
            save_plot_art_14(
                [y_enc_14.T],
                f"{log_dir}/generated_enc_{i}.png",
            )
            save_plot_art_14(
                [y_dec_14.T],
                f"{log_dir}/generated_dec_{i}.png",
            )
            save_plot(
                attn.squeeze().cpu(),
                f"{log_dir}/alignment_{i}.png",
            )
        logger.add_scalar(
            "valid_batch/dtw_enc_score",
            np.mean(gt_enc_dtw_scores),
            global_step=epoch,
        )

        logger.add_scalar(
            "valid_batch/dtw_dec_score",
            np.mean(gt_dec_dtw_scores),
            global_step=epoch,
        )


def train():
    mylogger.info("Initializing logger...")
    logger = SummaryWriter(log_dir=params_v6.log_dir)

    mylogger.info("Initializing data loaders...")
    fd = FeatureDecoder(sum_diphthong=True)
    tokenizer = FeatureTokenizer(fd)

    train_dataset = get_train_dataset(tokenizer)
    val_dataset = get_valid_dataset(tokenizer)

    train_loader = init_data_loader(
        train_dataset,
        validation=False,
        multilingual=params_v6.separate_files,
    )
    val_loader = init_data_loader(
        val_dataset,
        validation=True,
        multilingual=params_v6.separate_files,
    )
    mylogger.info("Train dataset initialized with %d samples", len(train_dataset))
    mylogger.info("Valid dataset initialized with %d samples", len(val_dataset))
    mylogger.info("Defining and logging valid batch...")
    valid_batch = log_valid_batch(logger, val_dataset)

    mylogger.info("Initializing model...")

    model = init_model()

    mylogger.info("Model initialized.")

    mylogger.info("Number of encoder parameters: %.2fm" % (model.encoder.nparams / 1e6))
    mylogger.info("Number of decoder parameters: %.2fm" % (model.decoder.nparams / 1e6))
    mylogger.info("Total parameters: %.2fm" % (model.nparams / 1e6))

    if start_epoch == 1:
        mylogger.info(
            f"Loading model state dict from ckpt grad_{start_epoch - 1}.pt ..."
        )

    else:  # continue training from a checkpoint
        mylogger.info(
            f"Loading model state dict from ckpt grad_{start_epoch - 1}.pt ..."
        )
        ckpt_grad = torch.load(
            log_dir / f"grad_{start_epoch - 1}.pt",
            weights_only=True,
        )
        model.load_state_dict(ckpt_grad)
        mylogger.info("Model state dict loaded.")

    mylogger.info("Loading optimizer...")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params_v6.learning_rate)

    mylogger.info("Start training...")
    progress_bar = tqdm(
        range(start_epoch, end_epoch + 1),
        total=end_epoch - start_epoch + 1,
        desc="Training",
        position=1,
        dynamic_ncols=True,
    )
    for epoch in range(start_epoch, end_epoch + 1):
        progress_bar.set_description(f"Training Epoch {epoch}")
        avg_losses, max_grad_norms = train_loop(model, train_loader, optimizer, epoch)

        mylogger.info(f"Epoch {epoch} finished")
        # Save epoch training losses to TensorBoard
        log_msg = save_train_scalars(logger, epoch, avg_losses, max_grad_norms)
        mylogger.info(f"Train : {log_msg}")
        # Evaluate validation loss
        if epoch % val_every == 0:
            mylogger.info(f"Validation loss at epoch {epoch}...")
            model.eval()
            val_losses = validation_loop(model, val_loader)
            log_msg = save_val_scalars(logger, epoch, val_losses)
            mylogger.info(f"Val : {log_msg}")

        # Save model every `save_every` epochs
        if epoch % save_every == 0:
            model.eval()
            mylogger.info("Synthesis...")
            save_synthesis(logger, model, epoch, valid_batch)
            mylogger.info("Synthesis finished.")

            ckpt = model.state_dict()
            torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")

        # manually update the progress bar
        progress_bar.update(1)

    mylogger.info("Training finished")
    mylogger.info("Saving final model...")
    ckpt = model.state_dict()
    torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
    mylogger.info("Final model saved.")


if __name__ == "__main__":
    torch.manual_seed(params_v6.random_seed)
    np.random.seed(params_v6.random_seed)

    train()
