# -*- coding: utf-8 -*-

"""EMA dataset objects

To structure EMA dataset processing
"""

from pathlib import Path
import sys

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import soundfile as sf
import joblib
import re

from utils_ema.cst import (
    SRC_DIR,
    MSPKA_ema_idx_to_keep,
    pb2007_idx_to_keep,
    pb2007_id2type,
    pb2007_ids_per_type,
    mochatimit_idx_to_keep,
)
from scipy.stats import pearsonr

# Set the random seed for reproducibility
seed = np.random.seed(42)
rng = np.random.default_rng(seed)


class SentenceMetadata:
    def __init__(
        self,
        id: int,
        filestem: str,
        sentence: str,
        duration: float = None,
        valid: bool = None,
        phone_fp: Path = None,
        sentence_type: str = None,
    ):
        self.id = id
        self.filestem = filestem
        self.sentence = sentence
        self.duration = duration
        self.valid = valid  # True if no NaN values in EMA data
        self.phone_fp = phone_fp
        self.split = None
        self.split_locs = None
        self.PCC_score = None

        self.sentence_type = sentence_type  # for pb2007

    def __repr__(self):
        return f"SentenceMetadata(sentence={self.sentence[:10]}, valid={self.valid})"

    def set_valid(self, valid: bool):
        self.valid = valid

    def set_duration(self, duration: float):
        self.duration = duration


class SpeakerMetadata:
    def __init__(
        self,
        speaker: str,
        datasets_dir: Path,
        dataset_name: str = "MSPKA_EMA_ita",
        target_sr: int = 100,
    ):
        self.speaker = speaker
        self.dataset_name = dataset_name

        with open(SRC_DIR / f"config_ema/{dataset_name}.yaml") as f:
            config = yaml.safe_load(f)

        self.config = config  # dataset config params

        # params
        self.audio_sr = config["audio_sr"] if "audio_sr" in config else None
        self.ema_sr = config["ema_sr"] if "ema_sr" in config else None
        self.target_sr = target_sr

        # directories
        self.datasets_dir = datasets_dir
        self.src_data_dir = datasets_dir / f"{dataset_name}/src_data"

        def get_path(rel_key: str):
            """Replace the speaker# tag in the subdir scheme with the actual speaker name
            for rel_dir_key in ["src_audio_reldir", "src_ema_reldir", "src_phone_reldir"]
            or rel_path_key in ["src_sentence_reldir"]
            """
            subdir = config[rel_key]
            return self.src_data_dir / subdir.replace("speaker#", f"{self.speaker}")

        self.src_audio_dir = (
            get_path("src_audio_reldir") if "src_audio_reldir" in config else None
        )
        self.src_ema_dir = (
            get_path("src_ema_reldir") if "src_ema_reldir" in config else None
        )
        self.src_phone_dir = (
            get_path("src_phone_reldir") if "src_phone_reldir" in config else None
        )
        self.src_sentence_dir = (
            get_path("src_sentence_reldir") if "src_sentence_reldir" in config else None
        )

        self.speaker_dir = (
            datasets_dir / f"{dataset_name}/processed_data/{speaker}"
        )  # speaker processed data dir

        # sentence metadata
        self.sentence_ids = []
        self.sentence_info = {}

        # Training info
        self.train_ids = []
        self.dev_ids = []
        self.test_ids = []

    def __repr__(self):
        return f"SpeakerMetadata(speaker={self.speaker}, dataset={self.dataset_name}, target_sr={self.target_sr})"

    def save(
        self,
        prefix: str = "",
    ):
        if prefix:  # to customize the file name
            filename = f"{prefix}_speaker_metadata_{self.target_sr}Hz.joblib"
        else:
            filename = f"speaker_metadata_{self.target_sr}Hz.joblib"
        joblib.dump(self, self.speaker_dir / filename)
        print(f"Saved speaker metadata to {self.speaker_dir / filename}")

    def load(
        self,
        prefix: str = "",
    ):
        if prefix:  # to customize the file name
            filename = f"{prefix}_speaker_metadata_{self.target_sr}Hz.joblib"
        else:
            filename = f"speaker_metadata_{self.target_sr}Hz.joblib"
        return joblib.load(self.speaker_dir / filename)

    def add_sentence(self, sentence_metadata: SentenceMetadata):
        self.sentence_ids.append(sentence_metadata.id)
        self.sentence_info[sentence_metadata.id] = sentence_metadata

    def list_valid_ids(self):
        return [id for id in self.sentence_ids if self.sentence_info[id].valid]

    def get_sentences(self):
        if self.dataset_name == "MSPKA_EMA_ita":
            return self.get_MSPKA_sentences()
        elif self.dataset_name == "pb2007":
            return self.get_pb2007_sentences()
        elif self.dataset_name == "mocha_timit":
            return self.get_mochatimit_sentences()
        else:
            raise ValueError(
                f"Dataset {self.dataset_name} not supported for sentence extraction"
            )

    def get_src_ema(self, id: int) -> np.ndarray:
        """
        Get the source EMA file for a given sentence
        """
        sentence_metadata = self.sentence_info[id]
        if self.dataset_name == "MSPKA_EMA_ita":
            return self.get_MSPKA_ema(sentence_metadata)
        elif self.dataset_name == "pb2007":
            return self.get_pb2007_ema(sentence_metadata)
        elif self.dataset_name == "mocha_timit":
            return self.get_mochatimit_ema(sentence_metadata)
        else:
            raise ValueError(
                f"Dataset {self.dataset_name} not supported for EMA extraction"
            )

    def extract_duration(self, id: int):
        sentence_metadata = self.sentence_info[id]
        if self.dataset_name == "MSPKA_EMA_ita":
            return self.extract_MSPKA_duration(sentence_metadata)
        elif self.dataset_name == "pb2007":
            return self.extract_pb2007_duration(sentence_metadata)
        elif self.dataset_name == "mocha_timit":
            return self.extract_mochatimit_duration(sentence_metadata)
        else:
            raise ValueError(
                f"Dataset {self.dataset_name} not supported for duration extraction"
            )

    def set_splits(
        self,
        split_arg: str,
    ):
        """
        Set the splits for the speaker
        """
        if self.dataset_name == "MSPKA_EMA_ita":
            train_ids, dev_ids, test_ids = self.set_MSPKA_splits(split_arg)
        elif self.dataset_name == "pb2007":
            train_ids, dev_ids, test_ids = self.set_pb2007_splits(split_arg)
        elif self.dataset_name == "mocha_timit":
            train_ids, dev_ids, test_ids = self.set_mochatimit_splits(split_arg)
        else:
            raise ValueError(
                f"Dataset {self.dataset_name} not supported for split setting"
            )
        # Save the split ids
        self.train_ids = train_ids
        self.dev_ids = dev_ids
        self.test_ids = test_ids
        return train_ids, dev_ids, test_ids

    def agg_Xy_split(
        self,
        ids: list,
        split: str,
    ):
        """
        Aggregate the design matrix and labels for a given set of sentences
        Set the split attribute and the location within the split for each sentence
        Args:
            ids (list): list of sentence ids
            target_sr (int): target sampling rate
            split (str): split name (train, dev, test)
        Returns:
            Xy (np.ndarray): design matrix and labels
        """
        Xy_dir = self.speaker_dir / "design_matrix"
        Xy_agg = []
        i = 0
        for id in ids:
            Xy = np.load(Xy_dir / f"Xy_{id}_{self.target_sr}Hz.npy")
            Xy_agg.append(Xy)
            self.sentence_info[id].split = split
            self.sentence_info[id].split_locs = (i, i + Xy.shape[0])
            i += Xy.shape[0]
        return np.concatenate(Xy_agg, axis=0)

    def compute_sentence_pcc(
        self,
        id: int,
        estimator,
        X_split: np.ndarray,
        y_split: np.ndarray,
        shift: int = 0,
    ):
        """
        Compute the Pearson correlation coefficient between the predicted
        and true values for a given sentence.
        """
        start, stop = self.sentence_info[id].split_locs
        X = X_split[shift + start : shift + stop, :]
        y = y_split[shift + start : shift + stop, :]
        y_pred = estimator.predict(X)
        corr, _ = pearsonr(y_pred, y)
        self.sentence_info[id].PCC_score = np.mean(corr)
        return corr

    # MSPKA specific methods
    def get_MSPKA_sentences(self):
        """
        Get the sentences for a given speaker
        """
        with open(self.src_data_dir / f"{self.speaker}_1.0.0/list_sentences") as f:
            lines = f.readlines()

        sentences = []
        for line in lines:
            parts = line.split(")")
            # Check if there are 2 parts
            if len(parts) == 2:
                id = int(parts[0].strip())
                sentence = parts[1].strip()
                filestem = self.config["filestem"].replace("speaker#", self.speaker)
                filestem = filestem.replace("id#", str(id))
                phone_fp = self.src_phone_dir / f"{filestem}.lab"
                sentences.append(
                    SentenceMetadata(
                        id=id,
                        sentence=sentence,
                        filestem=filestem,
                        phone_fp=phone_fp,
                    )
                )
            else:
                print("Line does not respect 'id) sentence' format:", line)
        return sentences

    def get_MSPKA_ema(self, sentence_metadata: SentenceMetadata):
        filestem = sentence_metadata.filestem
        src_ema_fp = self.src_ema_dir / f"{filestem}.ema"
        with open(src_ema_fp, "r") as f:
            lines = f.readlines()
        lines = [line.strip().split() for line in lines]

        ema = np.array(lines, dtype=np.float32)  # shape (n_channels, n_timesteps)
        ema = ema[MSPKA_ema_idx_to_keep, :].T  # shape (n_timesteps, n_channels)
        return ema

    def extract_MSPKA_duration(self, sentence_metadata: SentenceMetadata = None):
        """
        Extract the duration of a given sentence
        """
        try:
            # Get the duration from the lab file
            with open(sentence_metadata.phone_fp) as f:
                lines = f.readlines()
            # Get the duration from the lab file
            duration = lines[-1].split()[1]
            duration = float(duration)
            return duration
        except Exception as e:
            print(
                f"Error reading duration for {sentence_metadata.id} in {self.speaker}"
                f" from {sentence_metadata.phone_fp}: {e}"
            )
            return None

    def set_MSPKA_splits(
        self,
        split_arg: str,
    ):
        """
        Set the splits for the speaker
        Args:
            split_arg (str): split argument "nonmixed" or "mixed"
        """
        valid_ids = self.list_valid_ids()
        valid_ids = np.sort(valid_ids)
        train_size = int(0.65 * len(valid_ids))
        dev_size = int(0.15 * len(valid_ids))
        if split_arg == "mixed":
            # Shuffle the train, dev and test sets
            shuffled_ids = rng.permutation(valid_ids)
            train_ids, dev_ids, test_ids = np.split(
                shuffled_ids, [train_size, train_size + dev_size]
            )
        elif split_arg == "nonmixed":
            train_ids, dev_ids, test_ids = np.split(
                valid_ids, [train_size, train_size + dev_size]
            )
        else:
            print(
                f"Invalid split argument '{split_arg}' for MSPKA dataset. Using default mixed split"
            )
            shuffled_ids = rng.permutation(valid_ids)
            train_ids, dev_ids, test_ids = np.split(
                shuffled_ids, [train_size, train_size + dev_size]
            )
        return train_ids, dev_ids, test_ids

    # pb2007 specific methods
    def get_pb2007_sentences(self):
        """
        Get the sentences for a given speaker
        """
        phone_files = [f for f in self.src_phone_dir.glob("*.phone")]

        sentences = []
        for phone_fp in phone_files:
            filestem = phone_fp.stem
            id = int(filestem.split("_")[-1])
            sentence_type = pb2007_id2type[id]
            with open(phone_fp, "r") as file:
                lines = file.readlines()
                lines = [line.strip().split(" ") for line in lines]
                content = [line[-1] for line in lines if len(line) > 0]
                sentence = "".join(content)
                sentences.append(
                    SentenceMetadata(
                        id=id,
                        sentence=sentence,
                        filestem=filestem,
                        phone_fp=phone_fp,
                        sentence_type=sentence_type,
                    )
                )
        return sentences

    def get_pb2007_ema(self, sentence_metadata: SentenceMetadata):
        filestem = sentence_metadata.filestem
        src_ema_fp = self.src_ema_dir / f"{filestem}.ema"
        ema = np.fromfile(src_ema_fp, dtype=np.float32)
        ema = ema.reshape((-1, 12))  # shape (n_timesteps, n_channels)
        ema = ema[:, pb2007_idx_to_keep]
        return ema

    def extract_pb2007_duration(self, sentence_metadata: SentenceMetadata = None):
        file_id = str(sentence_metadata.id).zfill(4)
        wav, sr = sf.read(self.src_audio_dir / f"item_{file_id}.wav")
        return float(len(wav) / sr)

    def set_pb2007_splits(
        self,
        split_arg: str = "0.7",
    ):
        """
        Set the splits for the speaker
        Reminder : vowel: 39, vcv: 565, mono: 395, sentence: 110
        Default add vowel, vcv to train,
        splitmono to dev and sentence to test
        (v+vcv) / total = 0.54
        (v+vcv+0.3*mono) / total = 0.65
        (v+vcv+0.7*mono) / total = 0.8

        Args:
            split_arg (str): split argument. Default "0.7"
                            "float string" for example "0.1" share
                            of "mono" utt to add to train set)

                            OR alpha string indicating the types to include to train
                            ("v", "vcv", "m", "s") separated by & symbol
                            ordered as above for convenience
                            e.g. "v&vcv", "v&vcv&m", "v&vcv&m&s",
                            "s", "m&s", ...
        """
        valid_ids = self.list_valid_ids()
        v_ids = list(set(pb2007_ids_per_type["vowel"]).intersection(set(valid_ids)))
        vcv_ids = list(set(pb2007_ids_per_type["vcv"]).intersection(set(valid_ids)))
        mono_ids = list(set(pb2007_ids_per_type["mono"]).intersection(set(valid_ids)))
        sent_ids = list(
            set(pb2007_ids_per_type["sentence"]).intersection(set(valid_ids))
        )

        # check if alpha char in split_arg string
        if any(char.isalpha() for char in split_arg):
            expr = split_arg.split("&")  # parse the types to consider
            train_ids = []
            for code in expr:
                if code == "v":
                    train_ids += v_ids
                elif code == "vcv":
                    train_ids += vcv_ids
                elif code == "m":
                    train_ids += mono_ids
                elif code == "s":
                    train_ids += sent_ids
                else:
                    print(
                        f"Invalid split argument '{code} in {split_arg}' for pb2007 dataset."
                    )
                    pass
            test_ids = list(set(valid_ids) - set(train_ids))
            # divide train
            train_ids = rng.permutation(train_ids)  # shuffle train
            dev_split = int(0.8 * len(train_ids))
            dev_ids = train_ids[dev_split:]
            train_ids = train_ids[:dev_split]
            # shuffle train and dev
        else:  # float string for mono share
            train_ids = v_ids + vcv_ids  # default add vowel, vcv to train
            test_ids = sent_ids  # default add sentence to test

            # check split_arg format "x.x" using regex
            if bool(re.fullmatch(r"\d\.\d+", split_arg)):
                alpha = float(split_arg)  # share of mono to add to train+dev set
            else:
                print(
                    f"Invalid split argument '{split_arg}' for pb2007 dataset. Using default split."
                )
                alpha = 0.7

            # Affect randomly half of the alpha share to dev and half to train (1-alpha) to test
            start_test = int(alpha * len(mono_ids))
            start_dev = start_test // 2
            mono_ids_shuffled = rng.permutation(mono_ids)
            mono_ids_train, mono_ids_dev, mono_ids_test = np.split(
                mono_ids_shuffled, [start_dev, start_test]
            )

            train_ids += mono_ids_train.tolist()
            dev_ids = mono_ids_dev.tolist()
            test_ids += mono_ids_test.tolist()
        return np.array(train_ids), np.array(dev_ids), np.array(test_ids)

    # mochatimit specific methods

    def _read_mocha_ema(self, id):
        file_id = str(id).zfill(3)
        with open(self.src_ema_dir / f"{self.speaker}_{file_id}.ema", "rb") as f:
            # Read and parse header
            header_lines = []
            while True:
                line = f.readline().decode("ascii")
                header_lines.append(line)
                if line.strip() == "EST_Header_End":
                    break
            # Read rest of file as binary floats
            data = np.fromfile(f, dtype=np.float32)

        num_features = 22  # 1 time, 1 valid, 20 EMA values
        assert data.size % num_features == 0, (
            "Data does not align to expected frame size"
        )
        frames = data.reshape(-1, num_features)
        # parse into dictionary
        parsed = {
            "time": frames[:, 0],
            "valid": frames[:, 1],
            "ema": frames[:, 2:22],
            "header": header_lines,
        }
        return parsed

    def get_mochatimit_sentences(self):
        """
        Get the sentences for a given speaker
        """
        phone_files = [f for f in self.src_phone_dir.glob("*.phnm")]

        sentences = []
        for phone_fp in phone_files:
            filestem = phone_fp.stem
            sentence_fp = self.src_sentence_dir / f"{filestem}.trans"
            id = int(filestem.split("_")[-1])
            with open(sentence_fp, "r") as file:
                sentence = file.read()

            valid_list = self._read_mocha_ema(id)["valid"]
            if np.isnan(valid_list).any():
                continue
            else:
                sentences.append(
                    SentenceMetadata(
                        id=id,
                        sentence=sentence,
                        filestem=filestem,
                        phone_fp=phone_fp,
                        valid=True,
                    )
                )
        return sentences

    def get_mochatimit_ema(self, sentence_metadata: SentenceMetadata):
        ema = self._read_mocha_ema(sentence_metadata.id)[
            "ema"
        ]  # shape (n_timesteps, 20)
        ema = ema[:, mochatimit_idx_to_keep]  # shape (n_timesteps, 12)
        ema = ema.astype(np.float32)
        return ema

    def extract_mochatimit_duration(self, sentence_metadata: SentenceMetadata = None):
        file_id = str(sentence_metadata.id).zfill(3)
        wav, sr = sf.read(self.src_audio_dir / f"{self.speaker}_{file_id}.wav")
        return float(len(wav) / sr)

    def set_mochatimit_splits(
        self,
        split_arg: str = "0.7",
    ):
        """
        Set the splits for the speaker
        Args:
            split_arg (str): split argument "nonmixed" or "mixed"
        """
        valid_ids = self.list_valid_ids()
        valid_ids = np.sort(valid_ids)
        train_size = int(0.65 * len(valid_ids))
        dev_size = int(0.15 * len(valid_ids))
        if split_arg == "mixed":
            # Shuffle the train, dev and test sets
            shuffled_ids = rng.permutation(valid_ids)
            train_ids, dev_ids, test_ids = np.split(
                shuffled_ids, [train_size, train_size + dev_size]
            )
        elif split_arg == "nonmixed":
            train_ids, dev_ids, test_ids = np.split(
                valid_ids, [train_size, train_size + dev_size]
            )
        else:
            print(
                f"Invalid split argument '{split_arg}' for MSPKA dataset. Using default mixed split"
            )
            shuffled_ids = rng.permutation(valid_ids)
            train_ids, dev_ids, test_ids = np.split(
                shuffled_ids, [train_size, train_size + dev_size]
            )
        return train_ids, dev_ids, test_ids
