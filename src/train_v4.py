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

from configs import params_v4
from model import GradTTS
from data_textart import TextArtDataset, TextArtBatchCollate
from utils import (
    plot_tensor,
    save_plot,
    plot_art_14,
    save_plot_art_14,
    TqdmLoggingHandler,
    EarlyStopping,
)
from metrics import normalized_dtw_score
from text.symbols import symbols

log_dir = params_v4.log_dir
reorder_feats = params_v4.reorder_feats

# Setup logger
mylogger = logging.getLogger(__name__)
mylogger.setLevel(logging.INFO)

handler = TqdmLoggingHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
mylogger.addHandler(handler)

start_epoch = 1
end_epoch = 5000
custom_patience = 5000
val_every = params_v4.val_every
save_every = params_v4.save_every


if __name__ == "__main__":
    torch.manual_seed(params_v4.random_seed)
    np.random.seed(params_v4.random_seed)

    mylogger.info("Initializing logger...")
    logger = SummaryWriter(log_dir=params_v4.log_dir)

    mylogger.info("Initializing data loaders...")
    train_dataset = TextArtDataset(
        params_v4.train_filelist_path,
        cmudict_path=params_v4.cmudict_path,
        add_blank=params_v4.add_blank,
        data_root_dir=params_v4.data_root_dir,
        reorder_feats=params_v4.reorder_feats,
        pitch_idx=params_v4.pitch_idx,
        normalize_loudness=params_v4.normalize_loudness,
        loudness_idx=params_v4.loudness_idx,
        load_coder=False,
        sparc_ckpt_path=params_v4.sparc_ckpt_path,
    )
    batch_collate = TextArtBatchCollate()
    loader = DataLoader(
        dataset=train_dataset,
        batch_size=params_v4.batch_size,
        collate_fn=batch_collate,
        drop_last=True,
        num_workers=3,
        shuffle=False,
    )
    valid_dataset = TextArtDataset(
        params_v4.valid_filelist_path,
        cmudict_path=params_v4.cmudict_path,
        add_blank=params_v4.add_blank,
        data_root_dir=params_v4.data_root_dir,
        reorder_feats=params_v4.reorder_feats,
        pitch_idx=params_v4.pitch_idx,
        normalize_loudness=params_v4.normalize_loudness,
        loudness_idx=params_v4.loudness_idx,
        load_coder=False,
        sparc_ckpt_path=params_v4.sparc_ckpt_path,
    )
    val_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=params_v4.batch_size,
        collate_fn=batch_collate,
        drop_last=False,
        num_workers=3,
        shuffle=False,
    )

    mylogger.info("Initializing model...")

    add_blank = params_v4.add_blank
    nsymbols = len(symbols) + 1 if add_blank else len(symbols)

    model = GradTTS(
        nsymbols,
        params_v4.n_spks,
        None if params_v4.n_spks == 1 else params_v4.spk_embed_dim,
        params_v4.n_enc_channels,
        params_v4.filter_channels,
        params_v4.filter_channels_dp,
        params_v4.n_heads,
        params_v4.n_enc_layers,
        params_v4.enc_kernel,
        params_v4.enc_dropout,
        params_v4.window_size,
        params_v4.n_feats,
        params_v4.dec_dim,
        params_v4.beta_min,
        params_v4.beta_max,
        params_v4.pe_scale,
    ).cuda()

    mylogger.info("Model initialized.")

    # Check if we are continuing from a checkpoint or starting from scratch
    if start_epoch == 1:  # start training from scratch
        early_stopping = EarlyStopping(
            patience=custom_patience,
            step_size=val_every,
        )

    else:  # continue training from a checkpoint
        mylogger.info(f"Loading Early stopping from ckpt grad_{start_epoch - 1}.pt ...")
        early_stopping = torch.load(
            Path(params_v4.log_dir) / "early_stopping.pt", weights_only=False
        )

        mylogger.info(
            f"Loading model state dict from ckpt grad_{start_epoch - 1}.pt ..."
        )
        ckpt_grad = torch.load(Path(params_v4.log_dir) / f"grad_{start_epoch - 1}.pt")
        model.load_state_dict(ckpt_grad)
        mylogger.info("Model state dict loaded.")

    mylogger.info(
        "Number of encoder + duration predictor parameters: %.2fm"
        % (model.encoder.nparams / 1e6)
    )
    mylogger.info("Number of decoder parameters: %.2fm" % (model.decoder.nparams / 1e6))
    mylogger.info("Total parameters: %.2fm" % (model.nparams / 1e6))

    mylogger.info("Initializing optimizer...")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params_v4.learning_rate)

    mylogger.info("Logging valid batch...")
    valid_batch = valid_dataset.sample_test_batch(size=params_v4.test_size)
    for i, item in enumerate(valid_batch):
        art = item["y"][reorder_feats, :].cpu()
        logger.add_image(
            f"image_{i}/ground_truth",
            plot_art_14([art])[1],
            global_step=0,
            dataformats="HWC",
        )
        save_plot_art_14([art], f"{params_v4.log_dir}/original_{i}.png")

    mylogger.info("Start training...")
    iteration = 0
    best_epoch = 0
    # with tqdm(
    #    range(1, params_v4.n_epochs + 1),
    #    total=params_v4.n_epochs,
    #    desc="Training",
    #    position=1,
    #    dynamic_ncols=True,
    # ) as progress_bar:
    with tqdm(
        range(start_epoch, end_epoch + 1),
        total=end_epoch - start_epoch + 1,
        desc="Training",
        position=1,
        dynamic_ncols=True,
    ) as progress_bar:
        for epoch in progress_bar:
            model.train()
            dur_losses = []
            prior_losses = []
            diff_losses = []
            losses = []
            enc_grad_norms = []
            dec_grad_norms = []
            for batch_idx, batch in enumerate(loader):
                optimizer.zero_grad()
                x, x_lengths = batch["x"].cuda(), batch["x_lengths"].cuda()
                y, y_lengths = batch["y"].cuda(), batch["y_lengths"].cuda()
                dur_loss, prior_loss, diff_loss = model.compute_loss(
                    x, x_lengths, y, y_lengths, out_size=params_v4.out_size
                )
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.encoder.parameters(), max_norm=1
                )
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.decoder.parameters(), max_norm=1
                )
                optimizer.step()

                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                losses.append(loss.item())
                enc_grad_norms.append(enc_grad_norm.cpu().numpy())
                dec_grad_norms.append(dec_grad_norm.cpu().numpy())

                # if batch_idx % 10 == 0:
                #    msg = f"Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}"
                #    progress_bar.set_description(msg)

                iteration += 1

            mean_train_dur_loss = np.mean(dur_losses)
            mean_train_prior_loss = np.mean(prior_losses)
            mean_train_diff_loss = np.mean(diff_losses)
            mean_train_loss = np.mean(losses)
            max_enc_grad_norm = np.max(enc_grad_norms)
            max_dec_grad_norm = np.max(dec_grad_norms)

            logger.add_scalar(
                "training/duration_loss", mean_train_dur_loss, global_step=epoch
            )
            logger.add_scalar(
                "training/prior_loss", mean_train_prior_loss, global_step=epoch
            )
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

            log_msg = "Epoch %d: duration loss = %.3f " % (epoch, np.mean(dur_losses))
            log_msg += "| prior loss = %.3f " % np.mean(prior_losses)
            log_msg += "| diffusion loss = %.3f\n" % np.mean(diff_losses)
            log_msg += "| loss = %.3f\n" % np.mean(losses)
            with open(f"{log_dir}/train.log", "a") as f:
                f.write(log_msg)
            mylogger.info(f"Train : {log_msg}")

            # Evaluate validation loss
            if epoch % val_every == 0:
                mylogger.info(f"Validation loss at epoch {epoch}...")
                model.eval()
                val_dur_losses = []
                val_prior_losses = []
                val_diff_losses = []
                val_losses = []
                for batch_idx, batch in enumerate(val_loader):
                    with torch.no_grad():
                        x, x_lengths = batch["x"].cuda(), batch["x_lengths"].cuda()
                        y, y_lengths = batch["y"].cuda(), batch["y_lengths"].cuda()
                        dur_loss, prior_loss, diff_loss = model.compute_loss(
                            x, x_lengths, y, y_lengths, out_size=params_v4.out_size
                        )
                        val_loss = sum([dur_loss, prior_loss, diff_loss])

                        val_dur_losses.append(dur_loss.item())
                        val_prior_losses.append(prior_loss.item())
                        val_diff_losses.append(diff_loss.item())
                        val_losses.append(val_loss.item())

                mean_val_dur_loss = np.mean(val_dur_losses)
                mean_val_prior_loss = np.mean(val_prior_losses)
                mean_val_diff_loss = np.mean(val_diff_losses)
                mean_val_loss = np.mean(val_losses)

                logger.add_scalar(
                    "validation/duration_loss",
                    mean_val_dur_loss,
                    global_step=epoch,
                )
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
                log_msg = "Epoch %d: duration loss = %.3f " % (
                    epoch,
                    mean_val_dur_loss,
                )
                log_msg += "| prior loss = %.3f " % mean_val_prior_loss
                log_msg += "| diffusion loss = %.3f\n" % mean_val_diff_loss
                log_msg += "| loss = %.3f\n" % mean_val_loss
                with open(f"{log_dir}/val.log", "a") as f:
                    f.write(log_msg)
                mylogger.info(f"Val : {log_msg}")

                patience_counter, glob_improv = early_stopping.step(
                    [
                        mean_val_prior_loss,
                        mean_val_diff_loss,
                        mean_val_dur_loss,
                        mean_val_loss,
                    ]
                )

                mylogger.info(f"patience_counter: {patience_counter}")

                # save early stopping object
                torch.save(
                    early_stopping,
                    f=f"{log_dir}/early_stopping.pt",
                )

                if glob_improv:
                    torch.save(model.state_dict(), f=f"{log_dir}/grad_best.pt")
                    best_epoch = epoch
                    mylogger.info(
                        f"Best model saved at epoch {best_epoch} with validation loss {mean_val_loss:.3f}"
                    )
                # elif patience_counter >= params_v4.patience:
                elif patience_counter >= custom_patience:
                    mylogger.info(
                        f"Early stopping at epoch {epoch} after {early_stopping.counter} times {save_every} epochs without \
                            any subloss improvement. Best model epoch: {best_epoch}"
                    )
                    break

            # Save model every `save_every` epochs
            if epoch % save_every == 0:
                model.eval()
                mylogger.info("Synthesis...")
                gt_enc_dtw_scores = []
                gt_dec_dtw_scores = []
                with torch.no_grad():
                    for i, item in enumerate(valid_batch):
                        x = item["x"].to(torch.long).unsqueeze(0).cuda()
                        x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                        y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=50)

                        y_enc_14 = y_enc[
                            0, reorder_feats, :
                        ].T.cpu()  # (n_frames, n_feats)
                        y_dec_14 = y_dec[
                            0, reorder_feats, :
                        ].T.cpu()  # (n_frames, n_feats)
                        y_gt = (
                            item["y"][reorder_feats, :].T.cpu().numpy()
                        )  # (n_frames, n_feats)

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
                ckpt = model.state_dict()
                torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")

    ckpt = model.state_dict()
    torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
