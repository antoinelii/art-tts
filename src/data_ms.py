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

from pathlib import Path

from utils import normalize_channel

# from model.utils import fix_len_compatibility
from configs.params_v6 import random_seed
from configs.params_v6 import (
    reorder_feats,
    pitch_idx,
    loudness_idx,
    log_normalize_loudness,
)
from model.utils import fix_len_compatibility

from voxcommunis.data import PanPhonInventory, FeatureTokenizer
from voxcommunis.io import read_alignment, read_manifest
from voxcommunis.utils import MyPathLike, unique_consecutive


class PhnmArticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: MyPathLike,  # where the split dir is located ()"train-1h" for instance)
        manifest_path: MyPathLike,
        alignment_path: MyPathLike,
        feature_tokenizer: FeatureTokenizer,
        separate_files: bool = False,
        reorder_feats=reorder_feats,
        pitch_idx=pitch_idx,
        loudness_idx=loudness_idx,
        log_normalize_loudness=log_normalize_loudness,
        random_seed=random_seed,
    ):
        super().__init__()
        self.feature_tokenizer = feature_tokenizer
        self.init_manifest_and_alignments(
            manifest_path, alignment_path, separate_files=separate_files
        )
        # articulatory settings
        self.dataset_dir = dataset_dir
        self.reorder_feats = reorder_feats
        self.pitch_idx = pitch_idx
        self.loudness_idx = loudness_idx
        self.log_normalize_loudness = log_normalize_loudness

        random.seed(random_seed)

    def init_manifest_and_alignments(
        self,
        manifest_path: MyPathLike,
        alignment_path: MyPathLike,
        separate_files: bool = False,
    ):
        panphon_inventory = PanPhonInventory()
        if separate_files:
            manifests_list = sorted(list(Path(manifest_path).glob("*.tsv")))
            self.langs = [fp.stem for fp in manifests_list]
            self.lang_sizes = []
            self.manifest = []
            for man_path in manifests_list:
                man = read_manifest(man_path)
                self.manifest += list(man.items())
                self.lang_sizes.append(len(man))

            self.ipa_phones: dict[str, str] = {}
            for lang in self.langs:
                align_path = Path(alignment_path) / f"{lang}.align"
                alignments = read_alignment(align_path)
                self.ipa_phones.update(
                    {
                        file: panphon_inventory.convert_to_ipa(_align)
                        for file, _align in alignments.items()
                    }
                )
        else:
            manifest = read_manifest(manifest_path)
            self.manifest = list(manifest.items())
            alignments = read_alignment(alignment_path)
            assert self.feature_tokenizer.multilingual_mode  # ???
            self.ipa_phones = {
                file: panphon_inventory.convert_to_ipa(_align)
                for file, _align in alignments.items()
            }

    def get_phon_feats(self, file_id: str) -> tuple[torch.Tensor]:
        phones = self.ipa_phones[file_id].split(" ")  # parsed as list of str
        phones, counts = unique_consecutive(phones, return_counts=True)
        counts1 = [1 for _ in phones]  # create counts of 1 for each phone
        phon_features, phones = self.feature_tokenizer.encode(phones, counts1)
        # Add silence trait as an additional 25th dimension
        sil_trait = (phon_features == 0).all(
            axis=1
        ) * 2 - 1  # 1 for sil, -1 for non-sil
        phon_features = torch.concat([phon_features, sil_trait.unsqueeze(1)], dim=1)
        # add counts to the as an additional 26th dimension
        counts = torch.as_tensor(counts, dtype=torch.float).unsqueeze(1)
        phon_features = torch.concat([phon_features, counts], dim=1)
        phon_features = torch.FloatTensor(phon_features).T
        return phon_features  # shape: (n_ipa_feats, seq_len)

    def get_art(
        self,
        file_id: str,
    ) -> torch.FloatTensor:  # shape: (n_art_feats, T)
        """
        Get articulatory features from filepaths.
        Args:
            file_id (str): sample uuid.
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

        lang = file_id.split("_")[2]
        encoded_dir = Path(self.dataset_dir) / "encoded_audio_multi" / lang
        art_fp = encoded_dir / "emasrc" / f"{file_id}.npy"

        if art_fp.exists():
            art = np.load(art_fp)[:, :14]
        else:
            raise FileNotFoundError(f"Preprocessed file {art_fp} does not exist.")

        # pad n_art_feats to 16
        art = reorder_art_feats(art)
        art = normalize_channel(art, channel_idx=self.pitch_idx)
        if self.log_normalize_loudness:
            art[:, self.loudness_idx] = np.log(art[:, self.loudness_idx] + 1e-9)
            art = normalize_channel(art, channel_idx=self.loudness_idx)
        return torch.FloatTensor(art).T  # shape: (n_art_feats, T)

    def get_spk_features(
        self,
        file_id: str,
    ) -> torch.FloatTensor:  # shape: (n_art_feats, T)
        """
        Get spk prembedding features
        Args:
            file_id (str): sample uuid.
        Returns:
            torch.FloatTensor: spk preemb features (shape: (1024,)).
        """

        lang = file_id.split("_")[2]
        encoded_dir = Path(self.dataset_dir) / "encoded_audio_multi" / lang
        spk_preemb_fp = encoded_dir / "spk_preemb" / f"{file_id}.npy"

        if spk_preemb_fp.exists():
            spk_preemb = np.load(spk_preemb_fp)
        else:
            raise FileNotFoundError(
                f"Preprocessed file {spk_preemb_fp} does not exist."
            )

        return torch.FloatTensor(spk_preemb)  # shape: (1024,)

    def __getitem__(self, index):
        if index >= len(self.manifest):
            raise IndexError(f"Index {index} out of range")

        file_id, (audio_path, num_samples) = self.manifest[index]
        phon_features = self.get_phon_feats(file_id)  # (n_ipa_feats, seq_len)
        art = self.get_art(file_id)  # (n_art_feats, T)
        spk_preemb = self.get_spk_features(file_id)  # (1024,)
        item = {
            "x": phon_features,
            "spk_ft": spk_preemb,
            "y": art,
        }
        return item

    def __len__(self):
        return len(self.manifest)

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

        spk_ft = torch.stack([item["spk_ft"] for item in batch], dim=0)
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
        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spk_ft": spk_ft,
        }


class PhnmDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: MyPathLike,  # where the split dir is located ()"train-1h" for instance)
        manifest_path: MyPathLike,
        alignment_path: MyPathLike,
        feature_tokenizer: FeatureTokenizer,
        separate_files: bool = False,
        reorder_feats=reorder_feats,
        pitch_idx=pitch_idx,
        loudness_idx=loudness_idx,
        log_normalize_loudness=log_normalize_loudness,
        random_seed=random_seed,
    ):
        super().__init__()
        self.feature_tokenizer = feature_tokenizer
        self.init_manifest_and_alignments(
            manifest_path, alignment_path, separate_files=separate_files
        )
        # articulatory settings
        self.dataset_dir = dataset_dir
        self.reorder_feats = reorder_feats
        self.pitch_idx = pitch_idx
        self.loudness_idx = loudness_idx
        self.log_normalize_loudness = log_normalize_loudness

        random.seed(random_seed)

    def init_manifest_and_alignments(
        self,
        manifest_path: MyPathLike,
        alignment_path: MyPathLike,
        separate_files: bool = False,
    ):
        panphon_inventory = PanPhonInventory()
        if separate_files:
            manifests_list = sorted(list(Path(manifest_path).glob("*.tsv")))
            self.langs = [fp.stem for fp in manifests_list]
            self.lang_sizes = []
            self.manifest = []
            for man_path in manifests_list:
                man = read_manifest(man_path)
                self.manifest += list(man.items())
                self.lang_sizes.append(len(man))

            self.ipa_phones: dict[str, str] = {}
            for lang in self.langs:
                align_path = Path(alignment_path) / f"{lang}.align"
                alignments = read_alignment(align_path)
                self.ipa_phones.update(
                    {
                        file: panphon_inventory.convert_to_ipa(_align)
                        for file, _align in alignments.items()
                    }
                )
        else:
            manifest = read_manifest(manifest_path)
            self.manifest = list(manifest.items())
            alignments = read_alignment(alignment_path)
            assert self.feature_tokenizer.multilingual_mode  # ???
            self.ipa_phones = {
                file: panphon_inventory.convert_to_ipa(_align)
                for file, _align in alignments.items()
            }

    def get_phon_feats(self, file_id: str) -> tuple[torch.Tensor]:
        phones = self.ipa_phones[file_id].split(" ")  # parsed as list of str
        phones, counts = unique_consecutive(phones, return_counts=True)
        counts1 = [1 for _ in phones]  # create counts of 1 for each phone
        phon_features, phones = self.feature_tokenizer.encode(phones, counts1)
        # Add silence trait as an additional 25th dimension
        sil_trait = (phon_features == 0).all(
            axis=1
        ) * 2 - 1  # 1 for sil, -1 for non-sil
        phon_features = torch.concat([phon_features, sil_trait.unsqueeze(1)], dim=1)
        # add counts to the as an additional 26th dimension
        counts = torch.as_tensor(counts, dtype=torch.float).unsqueeze(1)
        phon_features = torch.concat([phon_features, counts], dim=1)
        phon_features = torch.FloatTensor(phon_features).T
        return phon_features  # shape: (n_ipa_feats, seq_len)

    def get_spk_features(
        self,
        file_id: str,
    ) -> torch.FloatTensor:  # shape: (n_art_feats, T)
        """
        Get spk prembedding features
        Args:
            file_id (str): sample uuid.
        Returns:
            torch.FloatTensor: spk preemb features (shape: (1024,)).
        """

        lang = file_id.split("_")[2]
        encoded_dir = Path(self.dataset_dir) / "encoded_audio_multi" / lang
        spk_preemb_fp = encoded_dir / "spk_preemb" / f"{file_id}.npy"

        if spk_preemb_fp.exists():
            spk_preemb = np.load(spk_preemb_fp)
        else:
            raise FileNotFoundError(
                f"Preprocessed file {spk_preemb_fp} does not exist."
            )

        return torch.FloatTensor(spk_preemb)  # shape: (1024,)

    def __getitem__(self, index):
        if index >= len(self.manifest):
            raise IndexError(f"Index {index} out of range")

        file_id, (audio_path, num_samples) = self.manifest[index]
        phon_features = self.get_phon_feats(file_id)  # (n_ipa_feats, seq_len)
        spk_preemb = self.get_spk_features(file_id)  # (1024,)
        item = {
            "x": phon_features,
            "spk_ft": spk_preemb,
        }
        return item

    def __len__(self):
        return len(self.manifest)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class PhnmBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_ipa_feats = batch[0]["x"].shape[-2]

        spk_ft = torch.stack([item["spk_ft"] for item in batch], dim=0)
        x = torch.zeros((B, n_ipa_feats, x_max_length), dtype=torch.float32)
        x_lengths = []

        for i, item in enumerate(batch):
            x_ = item["x"]
            x_lengths.append(x_.shape[-1])
            x[i, :, : x_.shape[-1]] = x_

        x_lengths = torch.LongTensor(x_lengths)
        return {
            "x": x,
            "x_lengths": x_lengths,
            "spk_ft": spk_ft,
        }
