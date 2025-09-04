from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent))

import joblib
import pandas as pd
import numpy as np
import logging
import importlib

# from paths import DATA_DIR


from scipy.stats import pearsonr

from tqdm import tqdm

from utils_dataset.mspka import get_MSPKA_ema
from utils_dataset.pb2007 import get_pb2007_ema
from utils_dataset.mocha import get_mochatimit_ema
from utils_dataset.mngu0 import read_mngu0_ema

from voxcommunis.io import read_manifest
from utils import TqdmLoggingHandler

import argparse
import utils_ema.ema_dataset

useless = utils_ema.ema_dataset.SentenceMetadata(
    0, "", ""
)  # to avoid linting error from unused import that is needed

dataset_2_spkmetadata = {
    "MSPKA_EMA_ita": "mixed_speaker_metadata_100Hz.joblib",
    "pb2007": "1.0_speaker_metadata_100Hz.joblib",
    "mocha_timit": "mixed_speaker_metadata_100Hz.joblib",
}

dataset_2_linear_model = {
    "MSPKA_EMA_ita": "mixed_model_full_100Hz.joblib",
    "pb2007": "1.0_model_full_100Hz.joblib",
    "mocha_timit": "mixed_model_full_100Hz.joblib",
}

dataset_2_speakers = {
    "MSPKA_EMA_ita": ["cnz", "lls", "olm"],
    "pb2007": ["spk1"],
    "mocha_timit": ["faet0", "ffes0", "fsew0", "maps0", "mjjn0", "msak0"],
    "MNGU0": ["s1"],
}


def get_50Hz_ema(dataset, filepath):
    """
    Get the 50Hz EMA data for the specified dataset and speaker.
    """
    if dataset == "MSPKA_EMA_ita":
        return get_MSPKA_ema(filepath)[::8, :12]
    elif dataset == "pb2007":
        return get_pb2007_ema(filepath)[::2, :12]
    elif dataset == "mocha_timit":
        return get_mochatimit_ema(filepath)[::10, :12]
    elif dataset == "MNGU0":
        return read_mngu0_ema(filepath)[0][::4, :12]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def spk_EMA_transform(dataset, ema_data, spk_linearmodel):
    """
    Transforms sparc space ema data using speaker linear model
    (bridging from 'universal' sparc space to normalized speaker ground-truth)
    """
    # ema_data shape (T, 12)
    if dataset != "MNGU0":  # need to convert to speaker space
        ema_data = (ema_data - np.mean(ema_data, axis=0)) / np.std(
            ema_data, axis=0
        )  # normalize using global stats
        ema_data = spk_linearmodel.predict(ema_data)
    return ema_data


def get_spk_summary_df(dataset, speaker, processed_data_dir, spkmetadata_filename):
    # get the metadata for each sentence
    spkmeta = joblib.load(processed_data_dir / f"{speaker}/{spkmetadata_filename}")
    ids = spkmeta.list_valid_ids()
    # pcc_scores = []
    filestems = []
    splits = []
    durations = []
    for id in ids:
        sentencemeta = spkmeta.sentence_info[id]
        filestems.append(sentencemeta.filestem)
        splits.append(sentencemeta.split)
        # pcc_scores.append(sentencemeta.PCC_score)
        durations.append(sentencemeta.duration)

    summary_df = pd.DataFrame(
        {
            "filestem": filestems,
            "split": splits,
            # "pcc": pcc_scores,
            "duration": durations,
        }
    )
    if dataset == "pb2007":
        sentence_types = [spkmeta.sentence_info[id].sentence_type for id in ids]
        summary_df["sentence_types"] = sentence_types

    # summary_df = summary_df.sort_values(by="pcc", ascending=False)

    return summary_df


def get_spk_summary_df_mngu0(src_data_dir):
    filestems = []
    # pccs = []
    durations = []
    for raw_ema_fp in list(src_data_dir.glob("*.ema")):
        sample_id = raw_ema_fp.stem
        ema_data, nonan = read_mngu0_ema(raw_ema_fp)
        if nonan:  # and ema_data.shape[0] / 4 > 150:
            filestems.append(sample_id)
            durations.append(ema_data.shape[0] / 200)  # Original data 200 Hz
            # ema_data = (ema_data - ema_data.mean(axis=0)) / ema_data.std(axis=0)
            # ema_data = ema_data[::4, :]  # TO 50 Hz
            # sparc_ema = np.load(sparc_pred_dir / f"{sample_id}.npy")[:, :12]
            # phnm3 = np.load(phnm3_dir / f"{sample_id}_phnm3.npy")
            # pcc, _ = pearsonr(sparc_ema, ema_data[: sparc_ema.shape[0]])
            # pccs.append(pcc)
    # pccs = np.array(pccs)

    summary_df = pd.DataFrame(
        {
            "filestem": filestems,
            # "pcc": pccs.mean(axis=1),
            "duration": durations,
        }
    )
    # summary_df = summary_df.sort_values(by="pcc", ascending=False)
    return summary_df


def get_spk_dataset_info(main_data_dir, dataset, speaker):
    if dataset != "MNGU0":
        processed_data_dir = main_data_dir / dataset / "processed_data"
        spkmetadata_filename = dataset_2_spkmetadata[dataset]
        linearmodel_name = dataset_2_linear_model[dataset]
        summary_df = get_spk_summary_df(
            dataset, speaker, processed_data_dir, spkmetadata_filename
        )
        spk_linearmodel_dir = (
            main_data_dir / dataset / "processed_data" / speaker / "linear_models"
        )
        spk_linearmodel = joblib.load(spk_linearmodel_dir / linearmodel_name)
        spk_gt_ema_dir = main_data_dir / dataset / "src_data" / speaker
    else:  # MNGU0
        src_data_dir = main_data_dir / dataset / "src_data" / speaker / "ema_basic_data"
        summary_df = get_spk_summary_df_mngu0(src_data_dir)
        spk_gt_ema_dir = (
            main_data_dir / dataset / "src_data" / speaker / "ema_basic_data"
        )
        spk_linearmodel = None
    return summary_df, spk_gt_ema_dir, spk_linearmodel


def denormalize_pitch(sparc_art, pred_art):
    # renormalize pred_art using sparc stats input shapes: (T, 14)
    pitch_mu, pitch_std = (
        sparc_art[:, 12].mean(),
        sparc_art[:, 12].std(),
    )
    pred_art[:, 12] = pred_art[:, 12] * pitch_std + pitch_mu  # denormalize pitch
    return pred_art


def denormalize_loudness(sparc_art, pred_art):
    loudness_mu, loudness_std = (
        np.log(sparc_art[:, 13] + 1e-9).mean(),
        np.log(sparc_art[:, 13] + 1e-9).std(),
    )
    pred_art[:, 13] = (
        pred_art[:, 13] * loudness_std + loudness_mu
    )  # denormalize log-loudness
    pred_art[:, 13] = np.exp(pred_art[:, 13])  # delog log-loudness
    return pred_art


def match_arr_lens(arrs):
    axis = 0
    min_len = np.min([arr.shape[axis] for arr in arrs])
    arrs = [arr[:min_len] for arr in arrs]
    return arrs


mylogger = logging.getLogger(__name__)
mylogger.setLevel(logging.INFO)

handler = TqdmLoggingHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
mylogger.addHandler(handler)


parser = argparse.ArgumentParser()
# for ground truth data loading
parser.add_argument(
    "--main_data_dir",  # ex: ../../data
    type=str,
    help="Main data directory containing datasets",
)
parser.add_argument(
    "--dataset", type=str, default="MSPKA_EMA_ita", help="Dataset to process"
)
# for predicted data in Voxcom folder structure
parser.add_argument(
    "--preds_dir",  # ex: ../../data/VoxCommunis/MNGU0/arttts_pred/v6/grad_1000
    type=str,
)
parser.add_argument(
    "--save_dir",  # ex: ../../data/VoxCommunis/MNGU0/analysis
    type=str,
)
parser.add_argument(
    "--sparc_dir",
    type=str,
    default="",  # ex: ../../data/VoxCommunis/encoded_audio_multi/MNGU0
)
parser.add_argument(
    "--manifest_path",  # ex: ../../data/VoxCommunis/MNGU0/manifests/MNGU0.tsv
    type=str,
)
parser.add_argument(
    "--version",
    type=str,
    default="v6",
    help="Version of the model used to generate data",
)
parser.add_argument(
    "--ckpt_name",
    type=str,
    default="grad_1000",
    help="Checkpoint name for the model used to generate data",
)
parser.add_argument(
    "--params_name",
    type=str,
    default="params_v6",
    help="Parameters file name (without .py) used to generate data",
)
if __name__ == "__main__":
    args = parser.parse_args()
    main_data_dir = Path(args.main_data_dir)
    dataset = args.dataset

    preds_dir = Path(args.preds_dir)
    save_dir = Path(args.save_dir)
    manifest_path = Path(args.manifest_path)
    sparc_dir = Path(args.sparc_dir)

    params_name = args.params_name
    version = args.version
    ckpt_name = args.ckpt_name
    speakers = dataset_2_speakers[dataset]

    save_dir.mkdir(parents=True, exist_ok=True)
    params = importlib.import_module(f"configs.{params_name}")

    mylogger.info(f"Load {dataset} filelist...")
    manifest = read_manifest(manifest_path)
    manifest_ids = set(manifest.keys())
    dir_files = list(preds_dir.glob("*.npy"))
    dir_ids = set([f.stem for f in dir_files])
    file_ids = list(
        manifest_ids.intersection(dir_ids)
    )  # all the available files of the dataset

    res_df = None
    for speaker in speakers:
        mylogger.info(f"Getting info for dataset: {dataset}, speaker: {speaker}")
        spk_df, spk_gt_ema_dir, spk_linearmodel = get_spk_dataset_info(
            main_data_dir, dataset, speaker
        )

        # get the predicted EMAs from ART-TTS and SPARC
        # and add them to the previous table
        spk_file_ids = set(spk_df.filestem.tolist())
        spk_file_ids = spk_file_ids.intersection(set(file_ids))

        mylogger.info("Found %d samples to process", len(spk_file_ids))
        mylogger.info(
            "Lost %d samples from speaker metadata", len(spk_df) - len(spk_file_ids)
        )
        spk_df = spk_df[spk_df.filestem.isin(spk_file_ids)].reset_index(drop=True)

        data = []
        with tqdm(
            range(len(spk_df)),
            total=len(spk_df),
            desc="Inference",
            position=1,
            dynamic_ncols=True,
        ) as progress_bar:
            for i in progress_bar:
                row = spk_df.iloc[i]
                filestem = row["filestem"]
                pred_art = np.load(preds_dir / f"{filestem}.npy")
                assert len(pred_art.shape) == 2, (
                    f"Unexpected shape for pred_articulatory data: {pred_art.shape} != 2"
                )
                assert pred_art.shape[0] == 29, (
                    f"Unexpected shape for pred_articulatory data: {pred_art.shape[0]} != 29"
                )
                pred_art = pred_art[14:28, :].T  # take only decoder art (T,14)
                sparc_art = np.load(sparc_dir / f"emasrc/{filestem}.npy")[
                    :, :14
                ]  # (T, 14)

                # get normalized ground truth EMA
                gt_ema = get_50Hz_ema(dataset, spk_gt_ema_dir / f"{filestem}.ema")
                gt_ema = (gt_ema - np.mean(gt_ema, axis=0)) / np.std(
                    gt_ema, axis=0
                )  # (T, 12)

                # renormalize pred_art using sparc stats
                pred_art = denormalize_pitch(sparc_art, pred_art)
                if params.log_normalize_loudness:
                    pred_art = denormalize_loudness(sparc_art, pred_art)

                # Adjust lengths if needed
                pred_art, sparc_art, gt_ema = match_arr_lens(
                    [pred_art, sparc_art, gt_ema]
                )
                assert pred_art.shape == sparc_art.shape, (
                    f"Unexpected shapes after length matching: {pred_art.shape} != {sparc_art.shape}"
                )
                assert gt_ema.shape[0] == pred_art.shape[0], (
                    f"Unexpected shapes after length matching: {gt_ema.shape} != {pred_art.shape}"
                )

                # transform sparc_ema and pred_ema using speaker linear model
                # (bridging from speaker specific sparc space to normalized ground-truth)
                pred_art[:, :12] = spk_EMA_transform(
                    dataset, pred_art[:, :12], spk_linearmodel
                )
                sparc_art[:, :12] = spk_EMA_transform(
                    dataset, sparc_art[:, :12], spk_linearmodel
                )

                # Compute PCC correlations between the 3 pairs of EMA/articulatory data
                pearson_gt_sparc_ema, _ = pearsonr(sparc_art[:, :12], gt_ema)
                pearson_gt_dec_ema, _ = pearsonr(pred_art[:, :12], gt_ema)
                pearson_dec_sparc_ema, _ = pearsonr(pred_art[:, :12], sparc_art[:, :12])
                pearson_dec_sparc_pitch, _ = pearsonr(pred_art[:, 12], sparc_art[:, 12])
                pearson_dec_sparc_loudness, _ = pearsonr(
                    pred_art[:, 13], sparc_art[:, 13]
                )

                new_row = {
                    "sample_id": filestem,
                }
                new_row["pcc_gt_dec_ema"] = np.mean(pearson_gt_dec_ema[:12])
                new_row["pcc_gt_sparc_ema"] = np.mean(pearson_gt_sparc_ema[:12])
                new_row["pcc_sparc_dec_ema"] = np.mean(pearson_dec_sparc_ema[:12])
                new_row["pcc_sparc_dec_pitch"] = pearson_dec_sparc_pitch
                new_row["pcc_sparc_dec_loudness"] = pearson_dec_sparc_loudness
                data.append(new_row)

        spk_df = pd.DataFrame(data)

        if res_df is None:
            res_df = spk_df.copy()
        else:
            res_df = pd.concat([res_df, spk_df], ignore_index=True)

    res_df = pd.DataFrame(data)
    save_path = save_dir / f"quanti_gt_art_comp_{version}_{ckpt_name}.csv"
    # add it to existing csv if exists
    if save_path.exists():
        res_df_old = pd.read_csv(save_path)
        res_df = pd.concat([res_df_old, res_df], axis=0)
    res_df.to_csv(save_path, index=False)
