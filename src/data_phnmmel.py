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
import torchaudio as ta

from typing import List, Tuple
from pathlib import Path

from text.converters import ipa_to_ternary, diphtongues_ipa
from utils import parse_filelist

# from model.utils import fix_len_compatibility
from configs.params_v3 import random_seed
from configs.params_v3 import (
    data_root_dir,
    merge_diphtongues,
)
from model.utils import fix_len_compatibility

import sys

sys.path.insert(0, "hifi-gan")
from meldataset import mel_spectrogram


class PhnmMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        data_root_dir=data_root_dir,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        shuffle=True,
        random_seed=random_seed,
    ):
        self.filepaths_list = parse_filelist(filelist_path)
        self.data_root_dir = Path(data_root_dir)
        self.merge_diphtongues = (
            merge_diphtongues  # whether to merge diphthongs in IPA embedding
        )
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        random.seed(random_seed)
        if shuffle:
            random.shuffle(self.filepaths_list)

    def get_pair(
        self,
        filepaths: List[str],
    ) -> Tuple[
        torch.IntTensor, torch.FloatTensor
    ]:  # shape: (n_ipa_feats, seq_len), (n_mel_feats, T)
        """
        Get a pair of phoneme embedding and mel features from filepaths.
        Args:
            filepaths (List[str]): List of filepaths [wav_filepath, phnm3_filepath].
        Returns:
            Tuple[torch.IntTensor, torch.FloatTensor]: A tuple containing:
                - phoneme embedding (shape: (n_ipa_feats, seq_len))
                - mel features (shape: (n_mel_feats, T))
        """
        wav_filepath, phnm3_filepath = filepaths
        phnm_emb = self.get_phnm_emb(phnm3_filepath)
        mel = self.get_mel(wav_filepath)
        return (phnm_emb, mel)

    def get_mel(self, filepath, sample_rate=22050):
        wav_fp = filepath.replace("DUMMY/", str(self.data_root_dir) + "/")
        wav_fp = Path(wav_fp)
        if wav_fp.exists():
            audio, sr = ta.load(wav_fp)
            if sr != sample_rate:
                resampler = ta.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                audio = resampler(audio)
        else:
            raise FileNotFoundError(f"wav file {wav_fp} does not exist.")

        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()
        return mel

    def get_phnm_emb(
        self,
        phnm3_fp: str,
        remove_trailing_sil: bool = False,
    ) -> torch.IntTensor:  # shape: (n_ipa_feats, seq_len)
        # get ipa_phnm3 from file
        phnm3_fp = phnm3_fp.replace("DUMMY/", str(self.data_root_dir) + "/")
        ipa_phnm3 = np.load(phnm3_fp)  # shape: (seq_len, 3) where 3 = [start, end, ipa]
        if remove_trailing_sil:  # remove trailing silence
            while len(ipa_phnm3) > 0 and ipa_phnm3[0][2] == ".":
                ipa_phnm3 = ipa_phnm3[1:]
            while len(ipa_phnm3) > 0 and ipa_phnm3[-1][2] == ".":
                ipa_phnm3 = ipa_phnm3[:-1]
        ipawords_list = ["%".join([elem[2] for elem in ipa_phnm3])]
        ternary_phnm_emb = ipa_to_ternary(
            ipawords_list, merge_diphtongues=self.merge_diphtongues
        )
        ternary_phnm_emb = torch.FloatTensor(
            ternary_phnm_emb
        ).T  # shape: (n_ipa_feats, seq_len)
        return ternary_phnm_emb

    def get_x_durations(
        self, phnm3_fp: str, merge_diphtongues: bool = False
    ) -> np.ndarray:
        phnm3_fp = phnm3_fp.replace("DUMMY/", str(self.data_root_dir) + "/")
        phnm3 = np.load(phnm3_fp)
        if merge_diphtongues:
            durations = [e[1] - e[0] for e in phnm3]
        else:
            durations = []
            for start, end, phone in phnm3:
                if phone in diphtongues_ipa:
                    mid = (end + start) / 2
                    durations.append(mid - start)
                    durations.append(end - mid)
                else:
                    durations.append(end - start)
        frames_mult = [delta * 50 for delta in durations]  # 50 = Art feats rate
        frames_mult = torch.tensor(frames_mult, dtype=torch.float32)
        return frames_mult

    def __getitem__(self, index):
        text, art = self.get_pair(self.filepaths_list[index])
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


class PhnmMelBatchCollate(object):
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


class PhnmBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_ipa_feats = batch[0]["x"].shape[-2]

        x = torch.zeros((B, n_ipa_feats, x_max_length), dtype=torch.float32)
        x_lengths = []

        for i, item in enumerate(batch):
            x_ = item["x"]
            x_lengths.append(x_.shape[-1])
            x[i, :, : x_.shape[-1]] = x_

        x_lengths = torch.LongTensor(x_lengths)
        return {"x": x, "x_lengths": x_lengths}
