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

from text.converters import ipa_to_ternary
from utils import parse_filelist

# from model.utils import fix_len_compatibility
from configs.params_v1 import seed as random_seed
from configs.params_v1 import (
    data_root_dir,
    sparc_ckpt_path,
    reorder_feats,
    pitch_idx,
)
from model.utils import fix_len_compatibility


class PhnmArticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        data_root_dir=data_root_dir,
        reorder_feats=reorder_feats,
        pitch_idx=pitch_idx,
        merge_diphtongues=False,
        load_coder=False,
        sparc_ckpt_path=sparc_ckpt_path,
        shuffle=True,
        random_seed=random_seed,
    ):
        self.filepaths_list = parse_filelist(
            filelist_path
        )  # list of [wav_filepath, phnm3_filepath] fp relative to data_home_dir
        self.data_root_dir = Path(data_root_dir)
        # self.add_blank = add_blank  #no such thing as we have sequence of phonemes and not words
        self.sparc_ckpt_path = sparc_ckpt_path
        self.reorder_feats = reorder_feats
        self.pitch_idx = pitch_idx
        self.merge_diphtongues = (
            merge_diphtongues  # whether to merge diphthongs in IPA embedding
        )
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
        filepaths: List[str],
        from_preprocessed: bool = True,
    ) -> Tuple[
        torch.IntTensor, torch.FloatTensor
    ]:  # shape: (n_ipa_feats, seq_len), (n_art_feats, T)
        """
        Get a pair of phoneme embedding and articulatory features from filepaths.
        Args:
            filepaths (List[str]): List of filepaths [wav_filepath, phnm3_filepath].
            from_preprocessed (bool): Whether to load preprocessed features or not.
        Returns:
            Tuple[torch.IntTensor, torch.FloatTensor]: A tuple containing:
                - phoneme embedding (shape: (n_ipa_feats, seq_len))
                - articulatory features (shape: (n_art_feats, T))
        """
        phnm3_filepath = filepaths[1]
        phnm_emb = self.get_phnm_emb(phnm3_filepath)
        art = self.get_art(filepaths, from_preprocessed=from_preprocessed)
        return (phnm_emb, art)

    def get_phnm_emb(
        self,
        phnm3_fp: str,
    ) -> torch.IntTensor:  # shape: (n_ipa_feats, seq_len)
        # get ipa_phnm3 from file
        phnm3_fp = phnm3_fp.replace("DUMMY/", str(self.data_root_dir) + "/")
        ipa_phnm3 = np.load(phnm3_fp)
        ipawords_list = ["%".join([elem[2] for elem in ipa_phnm3])]
        ternary_phnm_emb = ipa_to_ternary(
            ipawords_list, merge_diphtongues=self.merge_diphtongues
        )
        ternary_phnm_emb = torch.FloatTensor(
            ternary_phnm_emb
        ).T  # shape: (n_ipa_feats, seq_len)
        return ternary_phnm_emb

    def get_art(
        self, filepaths: str, from_preprocessed: bool = True
    ) -> torch.FloatTensor:  # shape: (n_art_feats, T)
        """
        Get articulatory features from filepaths.
        Args:
            filepaths (str): List of filepaths [wav_filepath, phnm3_filepath].
            from_preprocessed (bool): Whether to load preprocessed features or not.
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

        def normalize_pitch_channel(art: np.ndarray) -> np.ndarray:
            """
            Normalize the pitch channel to have zero mean and unit variance.
            must be called after reordering the features.
            """
            std = np.std(art[:, self.pitch_idx])
            if std > 0:
                art[:, self.pitch_idx] = (
                    art[:, self.pitch_idx] - np.mean(art[:, self.pitch_idx])
                ) / np.std(art[:, self.pitch_idx])
            else:
                print("Zero variance in pitch channel. Centering to zero mean.")
                art[:, self.pitch_idx] = art[:, self.pitch_idx] - np.mean(
                    art[:, self.pitch_idx]
                )
            return art

        wav_fp, phnm3_fp = filepaths[0], filepaths[1]

        if from_preprocessed:  # Prefered way, loading precomputed features
            # Retrieve art fp from phnm3_fp
            phnm3_fp = phnm3_fp.replace("DUMMY/", str(self.data_root_dir) + "/")
            phnm3_stem = Path(phnm3_fp).stem
            artic_filename = f"{phnm3_stem[:-6]}.npy"  # remove "_phnm3" suffix
            artic_dir = Path(phnm3_fp).parent.parent / "encoded_audio_en"
            preprocessed_fp = artic_dir / "emasrc" / artic_filename
            if preprocessed_fp.exists():
                art = np.load(preprocessed_fp)[
                    :, :14
                ]  # Extract only the first 14 articulatory features
            else:
                raise FileNotFoundError(
                    f"Preprocessed file {preprocessed_fp} does not exist."
                )
        else:  # Long time, only use for live inference
            filepath = wav_fp.replace("DUMMY/", str(self.data_root_dir) + "/")
            if preprocessed_fp.exists():
                self.coder.eval()
                with torch.inference_mode():
                    outputs = self.coder.encode(filepath, concat=True)
                # Extract the first 14 features
                art = outputs["features"][:, :14]
            else:
                raise FileNotFoundError(
                    f"Preprocessed file {preprocessed_fp} does not exist."
                )
        # pad n_art_feats to 16
        art = reorder_art_feats(art)
        art = normalize_pitch_channel(art)
        return torch.FloatTensor(art).T  # shape: (n_art_feats, T)

    def __getitem__(self, index):
        text, art = self.get_pair(self.filepaths_list[index], from_preprocessed=True)
        item = {"y": art, "x": text}
        return item

    def __len__(self):
        return len(self.filepaths_list)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class PhnmArticBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_feats = batch[0]["y"].shape[-2]
        n_ipa_feats = batch[0]["x"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, n_ipa_feats, x_max_length), dtype=torch.float32)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, :, : x_.shape[-1]] = x_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {"x": x, "x_lengths": x_lengths, "y": y, "y_lengths": y_lengths}
