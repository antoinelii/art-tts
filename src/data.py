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

from text import cmudict
from text.converters import text_to_ipa, ipa_to_ternary
from utils import parse_filelist, intersperse

# from model.utils import fix_len_compatibility
from configs.params_v0 import seed as random_seed
from configs.params_v0 import wavs_dir, artic_dir, sparc_ckpt_path
from model.utils import fix_len_compatibility


device = "cuda" if torch.cuda.is_available() else "cpu"
spk_emb_save_dir = Path(artic_dir) / "spk_emb"
spk_emb_save_dir.mkdir(exist_ok=True)
ft_save_dir = Path(artic_dir) / "emasrc"
ft_save_dir.mkdir(exist_ok=True)
coder = load_model(ckpt=sparc_ckpt_path, device=device)


class TextArticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        cmudict_path,
        add_blank=True,
        sample_rate=22050,
    ):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank
        self.sample_rate = sample_rate
        random.seed(random_seed)
        random.shuffle(self.filepaths_and_text)

    def get_pair(
        self,
        filepath_and_text: List[str],
        from_preprocessed: bool = True,
    ) -> Tuple[
        torch.IntTensor, torch.FloatTensor
    ]:  # shape: (n_ipa_feats, seq_len), (n_art_feats, T)
        filepath, text = filepath_and_text[0], filepath_and_text[1]
        text = self.get_text(text, add_blank=self.add_blank)
        art = self.get_art(filepath, from_preprocessed=from_preprocessed)
        return (text, art)

    def get_text(
        self, text: str, add_blank: bool = True
    ) -> torch.IntTensor:  # shape: (n_ipa_feats, seq_len)
        ipawords_list = text_to_ipa(
            text,
            dictionary=self.cmudict,
            cleaner_names=["english_cleaners_v2"],
            remove_punctuation=False,
        )
        if add_blank:
            ipawords_list = intersperse(ipawords_list, " ")
        ternary_emb = ipa_to_ternary(ipawords_list)
        ternary_emb = torch.FloatTensor(ternary_emb).T  # shape: (n_ipa_feats, seq_len)
        return ternary_emb

    def get_art(
        self, filepath: str, from_preprocessed: bool = True
    ) -> torch.FloatTensor:  # shape: (n_art_feats, T)
        art_filename = f"{Path(filepath).stem}.npy"
        if from_preprocessed:  # Favor loading precomputed features
            preprocessed_fp = Path(artic_dir) / "emasrc" / art_filename
            if preprocessed_fp.exists():
                art = np.load(preprocessed_fp)[
                    :, :14
                ]  # Extract only the first 14 articulatory features
            else:
                raise FileNotFoundError(
                    f"Preprocessed file {preprocessed_fp} does not exist."
                )
        else:  # Long inference time better to precompute the features
            filepath = filepath.replace("DUMMY/", str(wavs_dir) + "/")
            with torch.inference_mode():
                outputs = coder.encode(filepath, concat=True)
            # Save the outputs to avoid recomputing
            if not ft_save_dir.exists():
                ft_save_dir.mkdir(parents=True, exist_ok=True)
            if not spk_emb_save_dir.exists():
                spk_emb_save_dir.mkdir(parents=True, exist_ok=True)
            ft_save_path = ft_save_dir / art_filename
            spk_emb_save_path = spk_emb_save_dir / art_filename
            np.save(ft_save_path, outputs["features"])
            np.save(spk_emb_save_path, outputs["spk_emb"])
            # Extract the first 14 features
            art = outputs["features"][:, :14]
        return torch.FloatTensor(art).T  # shape: (n_art_feats, T)

    def __getitem__(self, index):
        text, art = self.get_pair(
            self.filepaths_and_text[index], from_preprocessed=True
        )
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


class TextArticBatchCollate(object):
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
