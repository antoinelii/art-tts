# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import random
import numpy as np

import torch

from typing import List, Tuple
from pathlib import Path

from sparc import load_model

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import parse_filelist, intersperse, normalize_channel

from model.utils import fix_len_compatibility
from configs.params_v4 import random_seed
from configs.params_v4 import (
    data_root_dir,
    sparc_ckpt_path,
    reorder_feats,
    pitch_idx,
    normalize_loudness,
    loudness_idx,
    cmudict_path,
    add_blank,
)


class TextArtDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        cmudict_path=cmudict_path,
        data_root_dir=data_root_dir,
        reorder_feats=reorder_feats,
        pitch_idx=pitch_idx,
        normalize_loudness=normalize_loudness,
        loudness_idx=loudness_idx,
        load_coder=False,
        sparc_ckpt_path=sparc_ckpt_path,
        shuffle=True,
        random_seed=random_seed,
    ):
        self.filepaths_list = parse_filelist(
            filelist_path
        )  # list of [wav_filepath, phnm3_filepath] fp relative to data_home_dir
        # text
        self.data_root_dir = Path(data_root_dir)
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank
        # artic
        self.sparc_ckpt_path = sparc_ckpt_path
        self.reorder_feats = reorder_feats
        self.pitch_idx = pitch_idx
        normalize_loudness = normalize_loudness
        self.loudness_idx = loudness_idx

        random.seed(random_seed)
        if shuffle:
            random.shuffle(self.filepaths_list)
        # init SPARC model in case we need to extract features
        # only for live inference, otherwise use precomputed features
        if load_coder:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.coder = load_model(ckpt=sparc_ckpt_path, device=device)

    def get_pair(
        self,
        filepath_and_text: List[str],
    ) -> Tuple[
        torch.IntTensor, torch.FloatTensor
    ]:  # shape: (n_ipa_feats, seq_len), (n_art_feats, T)
        """
        Get a pair of text embedding and articulatory features from filepaths.
        Args:
            filepath_and_text (List[str]): List of filepaths [art_filepath, text].
        Returns:
            Tuple[torch.IntTensor, torch.FloatTensor]: A tuple containing:
                - text translated as symbols
                - articulatory features (shape: (n_art_feats, T))
        """
        art_fp, text = filepath_and_text[0], filepath_and_text[1]
        text = self.get_text(text, add_blank=self.add_blank)
        art = self.get_art(art_fp)
        return (text, art)

    def get_art(
        self,
        filepath: str,
    ) -> torch.FloatTensor:  # shape: (n_art_feats, T)
        """
        Get articulatory features from filepaths.
        Args:
            filepath (str): art features filepath
        Returns:
            torch.FloatTensor: Articulatory features (shape: (n_art_feats, T)).
        """

        def reorder_art_feats(art: np.ndarray) -> np.ndarray:
            """
            Fix the articulatory features to have 16 channels.
            The original SPARC model has 14 features, but we need to pad it to 16.
            We also reorder the features
            """
            art16 = np.zeros((art.shape[0], 16))
            art16 = np.zeros((art.shape[0], 16))
            for i, j in enumerate(self.reorder_feats):
                art16[:, j] = art[:, i]
            return art16

        art_fp = filepath.replace("DUMMY/", str(self.data_root_dir) + "/")
        if art_fp.exists():
            art = np.load(art_fp)[
                :, :14
            ]  # Extract only the first 14 articulatory features
        else:
            raise FileNotFoundError(f"Preprocessed file {art_fp} does not exist.")
        # pad n_art_feats to 16
        art = reorder_art_feats(art)
        art = normalize_channel(art, channel_idx=self.pitch_idx)
        if self.normalize_loudness:
            art = normalize_channel(art, channel_idx=self.loudness_idx)
        return torch.FloatTensor(art).T  # shape: (n_art_feats, T)

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(
                text_norm, len(symbols)
            )  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        text, art = self.get_pair(self.filepaths_and_text[index])
        item = {"y": art, "x": text}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextArtBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {"x": x, "x_lengths": x_lengths, "y": y, "y_lengths": y_lengths}


class TextBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        x_max_length = max([item["x"].shape[-1] for item in batch])

        x = torch.zeros((B, x_max_length), dtype=torch.long)
        x_lengths = []

        for i, item in enumerate(batch):
            x_ = item["x"]
            x_lengths.append(x_.shape[-1])
            x[i, : x_.shape[-1]] = x_

        x_lengths = torch.LongTensor(x_lengths)
        return {"x": x, "x_lengths": x_lengths}
