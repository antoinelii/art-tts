# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import logging

# for DDP
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from balance_batch import DistLengthGroupedSampler

# from torch.profiler import profile, record_function, ProfilerActivity, schedule
from configs import params_v1_1
from data_phnm import PhnmArticBatchCollate, PhnmArticDataset
from metrics import normalized_dtw_score
from model import ArtTTS
from utils import (
    EarlyStopping,
    TqdmLoggingHandler,
    plot_art_14,
    plot_tensor,
    save_plot,
    save_plot_art_14,
)

log_dir = params_v1_1.log_dir
# profile_log_dir = log_dir + "_profiler"
reorder_feats = params_v1_1.reorder_feats

# Setup logger
mylogger = logging.getLogger(__name__)
mylogger.setLevel(logging.INFO)

handler = TqdmLoggingHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
mylogger.addHandler(handler)

mylogger.info("Initializing logger...")  # TensorBoard logger
logger = SummaryWriter(log_dir=params_v1_1.log_dir)

start_epoch = 1
end_epoch = 200
custom_patience = 5000
val_every = params_v1_1.val_every
save_every = params_v1_1.save_every


def get_lengths(dataset):
    """Get lengths of training samples."""
    lengths = []
    for i in range(len(dataset.filepaths_list)):
        length = dataset.get_phnm_emb(phnm3_fp=dataset.filepaths_list[i][1]).shape[1]
        lengths.append(length)
    return lengths


# DDP
def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_train_dataset():
    """Initialize the training dataset."""
    train_dataset = PhnmArticDataset(
        params_v1_1.train_filelist_path,
        data_root_dir=params_v1_1.data_root_dir,
        reorder_feats=params_v1_1.reorder_feats,
        pitch_idx=params_v1_1.pitch_idx,
        loudness_idx=params_v1_1.loudness_idx,
        log_normalize_loudness=params_v1_1.log_normalize_loudness,
        merge_diphtongues=params_v1_1.merge_diphtongues,
        load_coder=False,
        sparc_ckpt_path=params_v1_1.sparc_ckpt_path,
        shuffle=params_v1_1.shuffle,  # do not shuffle for compatibility custom sampler
        random_seed=params_v1_1.random_seed,
    )
    return train_dataset


def get_valid_dataset():
    """Initialize the validation dataset."""
    valid_dataset = PhnmArticDataset(
        params_v1_1.valid_filelist_path,
        data_root_dir=params_v1_1.data_root_dir,
        reorder_feats=params_v1_1.reorder_feats,
        pitch_idx=params_v1_1.pitch_idx,
        loudness_idx=params_v1_1.loudness_idx,
        log_normalize_loudness=params_v1_1.log_normalize_loudness,
        merge_diphtongues=params_v1_1.merge_diphtongues,
        load_coder=False,
        sparc_ckpt_path=params_v1_1.sparc_ckpt_path,
        shuffle=False,  # do not shuffle for compatibility custom sampler
        random_seed=params_v1_1.random_seed,
    )
    return valid_dataset


def init_data_loader(dataset):
    lengths = get_lengths(dataset)
    sampler = DistLengthGroupedSampler(
        lengths=lengths,
        batch_size=params_v1_1.batch_size,
        generator=torch.manual_seed(params_v1_1.random_seed),
    )

    batch_collate = PhnmArticBatchCollate()
    loader = DataLoader(
        dataset=dataset,
        batch_size=params_v1_1.batch_size,
        collate_fn=batch_collate,
        drop_last=True,
        num_workers=3,
        shuffle=False,
        sampler=sampler,  # Use custom sampler for balanced batches
        pin_memory=True,
    )

    return loader


def log_valid_batch(valid_dataset):
    """done only on rank 0"""
    valid_batch = valid_dataset.sample_test_batch(size=params_v1_1.test_size)
    for i, item in enumerate(valid_batch):
        art = item["y"][reorder_feats, :].cpu()
        logger.add_image(
            f"image_{i}/ground_truth",
            plot_art_14([art])[1],
            global_step=0,
            dataformats="HWC",
        )
        save_plot_art_14([art], f"{params_v1_1.log_dir}/original_{i}.png")
    return valid_batch


def init_model(rank, device):
    """Initialize the model on the specified device."""

    model = ArtTTS(
        params_v1_1.n_ipa_feats,
        params_v1_1.n_spks,
        None,
        params_v1_1.n_enc_channels,
        params_v1_1.filter_channels,
        params_v1_1.filter_channels_dp,
        params_v1_1.n_heads,
        params_v1_1.n_enc_layers,
        params_v1_1.enc_kernel,
        params_v1_1.enc_dropout,
        params_v1_1.window_size,
        params_v1_1.n_feats,
        params_v1_1.dec_dim,
        params_v1_1.beta_min,
        params_v1_1.beta_max,
        params_v1_1.pe_scale,
    ).to(device)

    DDP(model, device_ids=[rank])

    mylogger.info("DDP Model initialized on rank %d", rank)
    return model


def load_model(model, start_epoch, version, rank):
    """Check if we are continuing from a checkpoint or starting from scratch"""
    if start_epoch == 1:  # start training from scratch
        early_stopping = EarlyStopping(
            patience=custom_patience,
            step_size=val_every,
        )

    else:  # continue training from a checkpoint
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        early_stopping = torch.load(
            Path(params_v1_1.log_dir) / "early_stopping.pt",
            map_location=map_location,
            weights_only=False,
        )
        ckpt_grad = torch.load(
            Path(params_v1_1.log_dir) / f"grad_{start_epoch - 1}.pt",
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(ckpt_grad)
        mylogger.info("Model state dict loaded.")
    return model, early_stopping


def train_loop(model, train_loader, optimizer, epoch, device):
    model.train()
    dur_losses = []
    prior_losses = []
    diff_losses = []
    losses = []
    enc_grad_norms = []
    dec_grad_norms = []

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        x, x_lengths = batch["x"].to(device), batch["x_lengths"].to(device)
        y, y_lengths = batch["y"].to(device), batch["y_lengths"].to(device)
        dur_loss, prior_loss, diff_loss = model.compute_loss(
            x, x_lengths, y, y_lengths, out_size=params_v1_1.out_size
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

    mean_train_dur_loss = np.mean(dur_losses)
    mean_train_prior_loss = np.mean(prior_losses)
    mean_train_diff_loss = np.mean(diff_losses)
    mean_train_loss = np.mean(losses)
    max_enc_grad_norm = np.max(enc_grad_norms)
    max_dec_grad_norm = np.max(dec_grad_norms)
    means_tensor = torch.tensor(
        [
            mean_train_dur_loss,
            mean_train_prior_loss,
            mean_train_diff_loss,
            mean_train_loss,
        ],
        device=device,
    )
    maxs_tensor = torch.tensor(
        [max_enc_grad_norm, max_dec_grad_norm],
        device=device,
    )
    return (
        means_tensor,
        maxs_tensor,
    )


def reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Reduces a tensor across all processes by averaging.
    Args:
        tensor: 0D or 1D tensor (loss value(s))
        world_size: total number of processes
    Returns:
        Averaged tensor (same on all processes)
    """
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor


def reduce_max(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Reduces a tensor across all processes by taking the maximum.
    Args:
        tensor: 0D or 1D tensor (loss value(s))
        world_size: total number of processes
    Returns:
        Maximum tensor (same on all processes)
    """
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def save_train_scalars(logger, epoch, avg_losses, max_grad_norms):
    """Save training scalars to TensorBoard.
    Args:
        logger: TensorBoard logger
        epoch: Current epoch number
        avg_losses: Tensor containing mean loss values
        max_grad_norms: Tensor containing maximum gradient norms
    """

    mean_train_dur_loss = avg_losses[0].detach().cpu().numpy()
    mean_train_prior_loss = avg_losses[1].detach().cpu().numpy()
    mean_train_diff_loss = avg_losses[2].detach().cpu().numpy()
    mean_train_loss = avg_losses[3].detach().cpu().numpy()

    max_enc_grad_norm = max_grad_norms[0].detach().cpu().numpy()
    max_dec_grad_norm = max_grad_norms[1].detach().cpu().numpy()

    logger.add_scalar("training/duration_loss", mean_train_dur_loss, global_step=epoch)
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

    log_msg = "Epoch %d: duration loss = %.3f " % (epoch, mean_train_dur_loss)
    log_msg += "| prior loss = %.3f " % mean_train_prior_loss
    log_msg += "| diffusion loss = %.3f\n" % mean_train_diff_loss
    log_msg += "| loss = %.3f\n" % mean_train_loss
    with open(f"{log_dir}/train.log", "a") as f:
        f.write(log_msg)
    return log_msg


def validation_loop(model, val_loader, device):
    val_dur_losses = []
    val_prior_losses = []
    val_diff_losses = []
    val_losses = []
    for batch_idx, batch in enumerate(val_loader):
        with torch.no_grad():
            x, x_lengths = batch["x"].to(device), batch["x_lengths"].to(device)
            y, y_lengths = batch["y"].to(device), batch["y_lengths"].to(device)
            dur_loss, prior_loss, diff_loss = model.compute_loss(
                x, x_lengths, y, y_lengths, out_size=params_v1_1.out_size
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
    return [
        mean_val_prior_loss,
        mean_val_diff_loss,
        mean_val_dur_loss,
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
    mean_val_dur_loss = val_losses[2]
    mean_val_loss = val_losses[3]

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
    return log_msg


def save_synthesis(model, device, epoch, valid_batch):
    """Run inference on a list of filepaths and save the results.
    Args:
        model: Trained model for inference
    """
    gt_enc_dtw_scores = []
    gt_dec_dtw_scores = []
    with torch.no_grad():
        for i, item in enumerate(valid_batch):
            x = item["x"].to(torch.float32).unsqueeze(0).to(device)
            x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
            y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=50)

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


def train(rank, world_size):
    """Main training function for distributed training.
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    ddp_setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    train_dataset = get_train_dataset()
    valid_dataset = get_valid_dataset()
    train_loader = init_data_loader(train_dataset)
    val_loader = init_data_loader(valid_dataset)

    if rank == 0:
        mylogger.info("Train dataset initialized with %d samples", len(train_dataset))
        mylogger.info("Valid dataset initialized with %d samples", len(valid_dataset))
        mylogger.info("Defininf and logging valid batch...")
        valid_batch = log_valid_batch()
        mylogger.info("Initializing models...")

    model = init_model(rank, device)
    mylogger.info("Initializing optimizer on rank %d...", rank)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params_v1_1.learning_rate
    )

    # wait for all processes to finish model initialization
    dist.barrier()

    if rank == 0:
        mylogger.info(
            "Number of encoder + duration predictor parameters: %.2fm"
            % (model.encoder.nparams / 1e6)
        )
        mylogger.info(
            "Number of decoder parameters: %.2fm" % (model.decoder.nparams / 1e6)
        )
        mylogger.info("Total parameters: %.2fm" % (model.nparams / 1e6))

        if start_epoch == 1:
            mylogger.info(
                f"Loading Early stopping from ckpt grad_{start_epoch - 1}.pt ..."
            )
            mylogger.info(
                f"Loading model state dict from ckpt grad_{start_epoch - 1}.pt ..."
            )

    model, early_stopping = load_model(model, start_epoch, params_v1_1.version, rank)

    # wait for all processes to finish loading model and early stopping
    dist.barrier()

    mylogger.info("Start training on device %d...", rank)
    best_epoch = 0
    with tqdm(
        range(start_epoch, end_epoch + 1),
        total=end_epoch - start_epoch + 1,
        desc="Training",
        position=1,
        dynamic_ncols=True,
    ) as progress_bar:
        for epoch in progress_bar:
            means_tensor, maxs_tensor = train_loop(
                model, train_loader, optimizer, epoch, device
            )

            dist.barrier()  # Ensure all processes complete the training step before logging

            avg_losses = reduce_mean(means_tensor, world_size)
            max_grad_norms = reduce_max(maxs_tensor, world_size)

            if rank == 0:
                mylogger.info(f"Epoch {epoch} finished")
                # Save epoch training losses to TensorBoard
                log_msg = save_train_scalars(logger, epoch, avg_losses, max_grad_norms)
                mylogger.info(f"Train : {log_msg}")

                # Evaluate validation loss
                if epoch % val_every == 0:
                    mylogger.info(f"Validation loss at epoch {epoch}...")
                    model.eval()
                    val_losses = validation_loop(model, val_loader, device)
                    log_msg = save_val_scalars(logger, epoch, val_losses)
                    mylogger.info(f"Val : {log_msg}")

                    patience_counter, glob_improv = early_stopping.step(
                        val_losses,
                    )

                    mylogger.info(f"patience_counter: {patience_counter}")

                    # save early stopping object
                    torch.save(
                        early_stopping,
                        f=f"{log_dir}/early_stopping.pt",
                    )

                    if glob_improv:
                        torch.save(
                            model.module.state_dict(), f=f"{log_dir}/grad_best.pt"
                        )
                        best_epoch = epoch
                        mylogger.info(
                            f"Best model saved at epoch {best_epoch} with validation loss {val_losses[-1]:.3f}"
                        )
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
                    save_synthesis(model, device, epoch, valid_batch)
                    mylogger.info("Synthesis finished.")

                    ckpt = model.module.state_dict()
                    torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")

    cleanup()  # Clean up DDP resources
    if rank == 0:
        mylogger.info(f"Training finished. Best epoch: {best_epoch}")
        mylogger.info("Saving final model...")
        torch.save(model.state_dict(), f=f"{log_dir}/grad_final.pt")
        mylogger.info("Final model saved.")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--world_size",  # Number of processes to run in parallel
    type=int,
)

if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(params_v1_1.random_seed)
    np.random.seed(params_v1_1.random_seed)

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    mp.spawn(
        train,
        args=(args.world_size,),
        nprocs=args.world_size,
        join=True,
    )
    mylogger.info("Training script finished.")
