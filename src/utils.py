# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

import torch

from configs.params_v0 import pitch_idx, plot_norm_pitch

channel_names = [
    "xul",
    "zul",
    "xll",
    "zll",
    "xli",
    "zli",
    "xtt",
    "ztt",
    "xtm",
    "ztm",
    "xtb",
    "ztb",
    "pitch",
    "loudness",
]


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def latest_checkpoint_path(dir_path, regex="grad_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def load_checkpoint(logdir, model, num=None):
    if num is None:
        model_path = latest_checkpoint_path(logdir, regex="grad_*.pt")
    else:
        model_path = os.path.join(logdir, f"grad_{num}.pt")
    print(f"Loading checkpoint {model_path}...")
    model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
    model.load_state_dict(model_dict, strict=False)
    return model


def save_figure_to_numpy(fig):
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    return data


def plot_tensor(tensor, norm_pitch=plot_norm_pitch):
    tensor_ = tensor.detach().numpy().copy()
    if norm_pitch:
        tensor_[pitch_idx] = (tensor_[pitch_idx] - tensor_[pitch_idx].mean()) / tensor_[
            pitch_idx
        ].std()
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor_, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath, norm_pitch=plot_norm_pitch):
    tensor_ = tensor.detach().numpy().copy()
    if norm_pitch:
        tensor_[pitch_idx] = (tensor_[pitch_idx] - tensor_[pitch_idx].mean()) / tensor_[
            pitch_idx
        ].std()
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor_, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # keeps progress bar clean
            self.flush()
        except Exception:
            self.handleError(record)


class EarlyStopping:
    def __init__(self, patience=10, step_size=5):
        """
        EarlyStopping when none of the sublosses improve for a given number of steps.

        Args:
            patience (int): Number of steps (not successive epochs) to wait before stopping.
            step_size (int): Number of epochs between each patience check.
            smoothing_factor (float): Smoothing factor for EMA (0 < smoothing_factor <= 1).
        """
        self.patience = patience
        self.step_size = step_size
        self.counter = 0
        self.losses = [
            None,
            None,
            None,
            None,
        ]  # dur_loss, prior_loss, diff_loss, tot_loss
        self.best_losses = [float("inf"), float("inf"), float("inf"), float("inf")]

    def step(self, losses):
        """
        Update the EMA and check if early stopping criteria are met.

        Args:
            loss (float): list of losses to consider for early stopping.
                          (dur_loss, prior_loss, diff_loss, tot_loss)

        Returns:
            tuple: (should_stop, improved)
                - should_stop (bool): Whether to stop training.
                - improved (bool): Whether the loss has improved.
        """
        self.losses = losses

        # Check for improvement
        improvements = [self.losses[i] < self.best_losses[i] for i in range(4)]
        if any(improvements):
            self.counter = 0
            for i in range(4):
                if improvements[i]:
                    self.best_losses[i] = self.losses[i]
            if improvements[-1]:
                glob_improv = True
        else:
            self.counter += 1
            glob_improv = False

        # Return stopping condition and best model saving
        return self.counter >= self.patience, glob_improv

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_loss = float("inf")
        self.loss = None


def plot_art_14(tensor_list, title="Tensor", figsize=(8, 5), sr=50):
    """
    Plot each of the 14 features in its own subplot for each tensor
    of the list
    Overlay all tensors in the same figure.

    Args:
        tensor_list: list of np.ndarray or torch.Tensor of shape (14, n_frames)
        title: title prefix for the figure
        figsize: size of the entire figure
    Returns:
        Matplotlib figure
    """

    fig, axes = plt.subplots(7, 2, figsize=figsize, dpi=100, sharey=True, sharex=True)
    axes = axes.flatten()

    for tensor in tensor_list:
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()

        assert tensor.shape[0] == 14, "Expected tensor shape (14, n_frames)"
        t = np.arange(tensor.shape[1]) / sr
        for i in range(14):
            axes[i].plot(t, tensor[i], alpha=0.9)
            # axes[i].set_title(f"Feature {i}")
            # axes[i].set_xlabel("Frames")
            # axes[i].set_ylabel("Value")
            axes[i].grid(True)
    fig.suptitle(f"{title}", fontsize=16)
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close(fig)
    return fig, data


def save_plot_art_14(tensor_list, savepath, title="Tensor", figsize=(8, 5), sr=50):
    fig, axes = plt.subplots(7, 2, figsize=figsize, dpi=100, sharey=True, sharex=True)
    axes = axes.flatten()

    for tensor in tensor_list:
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()

        assert tensor.shape[0] == 14, "Expected tensor shape (14, n_frames)"
        t = np.arange(tensor.shape[1]) / sr
        for i in range(14):
            axes[i].plot(t, tensor[i], alpha=0.9)
            # axes[i].set_title(f"Feature {i}")
            # axes[i].set_xlabel("Frames")
            # axes[i].set_ylabel("Value")
            axes[i].grid(True)
    fig.suptitle(f"{title}", fontsize=16)
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close(fig)
    return
