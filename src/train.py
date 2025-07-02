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

from configs import params_v0
from model import ArtTTS
from data import TextArticDataset, TextArticBatchCollate
from utils import plot_tensor, save_plot, TqdmLoggingHandler, EarlyStopping
# from text.symbols import symbols

patience = params_v0.patience

train_filelist_path = params_v0.train_filelist_path
valid_filelist_path = params_v0.valid_filelist_path
cmudict_path = params_v0.cmudict_path
add_blank = params_v0.add_blank
n_spks = params_v0.n_spks
n_feats = params_v0.n_feats
plot_norm_pitch = params_v0.plot_norm_pitch
merge_diphtongues = params_v0.merge_diphtongues

log_dir = params_v0.log_dir
n_epochs = params_v0.n_epochs
batch_size = params_v0.batch_size
out_size = params_v0.out_size
learning_rate = params_v0.learning_rate
random_seed = params_v0.seed

n_ipa_feats = params_v0.n_ipa_feats
n_enc_channels = params_v0.n_enc_channels
filter_channels = params_v0.filter_channels
filter_channels_dp = params_v0.filter_channels_dp
n_enc_layers = params_v0.n_enc_layers
enc_kernel = params_v0.enc_kernel
enc_dropout = params_v0.enc_dropout
n_heads = params_v0.n_heads
window_size = params_v0.window_size

dec_dim = params_v0.dec_dim
beta_min = params_v0.beta_min
beta_max = params_v0.beta_max
pe_scale = params_v0.pe_scale

# Setup logger
mylogger = logging.getLogger(__name__)
mylogger.setLevel(logging.INFO)

handler = TqdmLoggingHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
mylogger.addHandler(handler)

# Setup early stopping
early_stopping = EarlyStopping(
    patience=patience,
    step_size=params_v0.val_every,
    smoothing_factor=0.1,
)

if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    mylogger.info("Initializing logger...")
    logger = SummaryWriter(log_dir=log_dir)

    mylogger.info("Initializing data loaders...")
    train_dataset = TextArticDataset(
        train_filelist_path,
        cmudict_path,
        add_blank,
        merge_diphtongues=merge_diphtongues,
    )
    batch_collate = TextArticBatchCollate()
    loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=batch_collate,
        drop_last=True,
        num_workers=3,
        shuffle=False,
    )
    valid_dataset = TextArticDataset(
        valid_filelist_path,
        cmudict_path,
        add_blank,
        merge_diphtongues=merge_diphtongues,
    )
    val_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        collate_fn=batch_collate,
        drop_last=False,
        num_workers=3,
        shuffle=False,
    )

    mylogger.info("Initializing model...")
    model = ArtTTS(
        n_ipa_feats,
        n_spks,
        None,
        n_enc_channels,
        filter_channels,
        filter_channels_dp,
        n_heads,
        n_enc_layers,
        enc_kernel,
        enc_dropout,
        window_size,
        n_feats,
        dec_dim,
        beta_min,
        beta_max,
        pe_scale,
    ).cuda()
    mylogger.info(
        "Number of encoder + duration predictor parameters: %.2fm"
        % (model.encoder.nparams / 1e6)
    )
    mylogger.info("Number of decoder parameters: %.2fm" % (model.decoder.nparams / 1e6))
    mylogger.info("Total parameters: %.2fm" % (model.nparams / 1e6))

    mylogger.info("Initializing optimizer...")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    mylogger.info("Logging valid batch...")
    valid_batch = valid_dataset.sample_test_batch(size=params_v0.test_size)
    for i, item in enumerate(valid_batch):
        art = item["y"]
        logger.add_image(
            f"image_{i}/ground_truth",
            plot_tensor(art.squeeze(), norm_pitch=plot_norm_pitch),
            global_step=0,
            dataformats="HWC",
        )
        save_plot(art.squeeze(), f"{log_dir}/original_{i}.png")

    mylogger.info("Start training...")
    iteration = 0
    with tqdm(
        range(1, n_epochs + 1),
        total=n_epochs,
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
            for batch_idx, batch in enumerate(loader):
                model.zero_grad()
                x, x_lengths = batch["x"].cuda(), batch["x_lengths"].cuda()
                y, y_lengths = batch["y"].cuda(), batch["y_lengths"].cuda()
                dur_loss, prior_loss, diff_loss = model.compute_loss(
                    x, x_lengths, y, y_lengths, out_size=out_size
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

                logger.add_scalar(
                    "training/duration_loss", dur_loss.item(), global_step=iteration
                )
                logger.add_scalar(
                    "training/prior_loss", prior_loss.item(), global_step=iteration
                )
                logger.add_scalar(
                    "training/diffusion_loss", diff_loss.item(), global_step=iteration
                )
                logger.add_scalar("training/loss", loss.item(), global_step=iteration)
                logger.add_scalar(
                    "training/encoder_grad_norm", enc_grad_norm, global_step=iteration
                )
                logger.add_scalar(
                    "training/decoder_grad_norm", dec_grad_norm, global_step=iteration
                )

                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                losses.append(loss.item())

                # if batch_idx % 10 == 0:
                #    msg = f"Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}"
                #    progress_bar.set_description(msg)

                iteration += 1

            log_msg = "Epoch %d: duration loss = %.3f " % (epoch, np.mean(dur_losses))
            log_msg += "| prior loss = %.3f " % np.mean(prior_losses)
            log_msg += "| diffusion loss = %.3f\n" % np.mean(diff_losses)
            log_msg += "| loss = %.3f\n" % np.mean(losses)
            with open(f"{log_dir}/train.log", "a") as f:
                f.write(log_msg)
            mylogger.info(f"Train : {log_msg}")

            # Evaluate validation loss
            if epoch % params_v0.val_every == 0:
                mylogger.info(f"Validation loss at epoch {epoch}...")
                model.eval()
                val_dur_losses = []
                val_prior_losses = []
                val_diff_losses = []
                val_losses = []
                for batch_idx, batch in enumerate(loader):
                    with torch.no_grad():
                        x, x_lengths = batch["x"].cuda(), batch["x_lengths"].cuda()
                        y, y_lengths = batch["y"].cuda(), batch["y_lengths"].cuda()
                        dur_loss, prior_loss, diff_loss = model.compute_loss(
                            x, x_lengths, y, y_lengths, out_size=out_size
                        )
                        val_loss = sum([dur_loss, prior_loss, diff_loss])
                        logger.add_scalar(
                            "validation/duration_loss",
                            dur_loss.item(),
                            global_step=iteration,
                        )
                        logger.add_scalar(
                            "validation/prior_loss",
                            prior_loss.item(),
                            global_step=iteration,
                        )
                        logger.add_scalar(
                            "validation/diffusion_loss",
                            diff_loss.item(),
                            global_step=iteration,
                        )
                        logger.add_scalar(
                            "validation/loss", val_loss.item(), global_step=iteration
                        )
                        val_dur_losses.append(dur_loss.item())
                        val_prior_losses.append(prior_loss.item())
                        val_diff_losses.append(diff_loss.item())
                        val_losses.append(val_loss.item())
                log_msg = "Epoch %d: duration loss = %.3f " % (
                    epoch,
                    np.mean(val_dur_losses),
                )
                log_msg += "| prior loss = %.3f " % np.mean(val_prior_losses)
                log_msg += "| diffusion loss = %.3f\n" % np.mean(val_diff_losses)
                log_msg += "| loss = %.3f\n" % np.mean(val_losses)
                with open(f"{log_dir}/val.log", "a") as f:
                    f.write(log_msg)
                mylogger.info(f"Val : {log_msg}")

                mean_val_loss = np.mean(val_losses)
                should_stop, improved = early_stopping.step(mean_val_loss)

                if improved:
                    patience_counter = 0
                    torch.save(model.state_dict(), f=f"{log_dir}/grad_best.pt")
                    mylogger.info(
                        f"Best model saved at epoch {epoch} with validation loss {mean_val_loss:.3f} \
                        and ema loss {early_stopping.ema_loss:.3f}."
                    )
                elif should_stop:
                    mylogger.info(
                        f"Early stopping at epoch {epoch} after {early_stopping.counter} times {params_v0.save_every} epochs without improvement."
                    )
                    break

            # Save model every `save_every` epochs
            if epoch % params_v0.save_every == 0:
                model.eval()
                mylogger.info("Synthesis...")
                with torch.no_grad():
                    for i, item in enumerate(valid_batch):
                        x = item["x"].to(torch.float32).unsqueeze(0).cuda()
                        x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                        y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=50)
                        logger.add_image(
                            f"image_{i}/generated_enc",
                            plot_tensor(
                                y_enc.squeeze().cpu(), norm_pitch=plot_norm_pitch
                            ),
                            global_step=iteration,
                            dataformats="HWC",
                        )
                        logger.add_image(
                            f"image_{i}/generated_dec",
                            plot_tensor(
                                y_dec.squeeze().cpu(), norm_pitch=plot_norm_pitch
                            ),
                            global_step=iteration,
                            dataformats="HWC",
                        )
                        logger.add_image(
                            f"image_{i}/alignment",
                            plot_tensor(attn.squeeze().cpu(), norm_pitch=False),
                            global_step=iteration,
                            dataformats="HWC",
                        )
                        save_plot(
                            y_enc.squeeze().cpu(),
                            f"{log_dir}/generated_enc_{i}.png",
                            norm_pitch=plot_norm_pitch,
                        )
                        save_plot(
                            y_dec.squeeze().cpu(),
                            f"{log_dir}/generated_dec_{i}.png",
                            norm_pitch=plot_norm_pitch,
                        )
                        save_plot(
                            attn.squeeze().cpu(),
                            f"{log_dir}/alignment_{i}.png",
                            norm_pitch=False,
                        )

                ckpt = model.state_dict()
                torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
