from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent))

import joblib
import pandas as pd
import numpy as np

from paths import DATA_DIR


from scipy.stats import pearsonr
from metrics import normalized_dtw_score

from tqdm import tqdm

from utils_dataset.mspka import get_MSPKA_ema
from utils_dataset.pb2007 import get_pb2007_ema
from utils_dataset.mocha import get_mochatimit_ema
from utils_dataset.mngu0 import read_mngu0_ema

import argparse

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


def get_sparc_summary_df(dataset, speaker, processed_data_dir, spkmetadata_filename):
    # get the PCC scores of SPARC and other metadata for each sentence
    spkmeta = joblib.load(processed_data_dir / f"{speaker}/{spkmetadata_filename}")
    ids = spkmeta.list_valid_ids()
    pcc_scores = []
    filestems = []
    splits = []
    durations = []
    for id in ids:
        sentencemeta = spkmeta.sentence_info[id]
        filestems.append(sentencemeta.filestem)
        splits.append(sentencemeta.split)
        pcc_scores.append(sentencemeta.PCC_score)
        durations.append(sentencemeta.duration)

    summary_df = pd.DataFrame(
        {
            "filestem": filestems,
            "split": splits,
            "pcc": pcc_scores,
            "duration": durations,
        }
    )
    if dataset == "pb2007":
        sentence_types = [spkmeta.sentence_info[id].sentence_type for id in ids]
        summary_df["sentence_types"] = sentence_types

    summary_df = summary_df.sort_values(by="pcc", ascending=False)

    return summary_df


def get_sparc_summary_df_mngu0(src_data_dir, sparc_pred_dir):
    filestems = []
    pccs = []
    durations = []
    for raw_ema_fp in list(src_data_dir.glob("*.ema")):
        sample_id = raw_ema_fp.stem
        ema_data, nonan = read_mngu0_ema(raw_ema_fp)
        if nonan:  # and ema_data.shape[0] / 4 > 150:
            filestems.append(sample_id)
            durations.append(ema_data.shape[0] / 200)  # Original data 200 Hz
            ema_data = (ema_data - ema_data.mean(axis=0)) / ema_data.std(axis=0)
            ema_data = ema_data[::4, :]  # TO 50 Hz
            sparc_ema = np.load(sparc_pred_dir / f"{sample_id}.npy")[:, :12]
            # phnm3 = np.load(phnm3_dir / f"{sample_id}_phnm3.npy")
            pcc, _ = pearsonr(sparc_ema, ema_data[: sparc_ema.shape[0]])
            pccs.append(pcc)
    pccs = np.array(pccs)

    summary_df = pd.DataFrame(
        {"filestem": filestems, "pcc": pccs.mean(axis=1), "duration": durations}
    )
    summary_df = summary_df.sort_values(by="pcc", ascending=False)
    return summary_df


parser = argparse.ArgumentParser()
parser.add_argument(
    "--main_data_dir",
    type=str,
    default=str(DATA_DIR),
    help="Main data directory containing datasets",
)
parser.add_argument(
    "--dataset", type=str, default="MSPKA_EMA_ita", help="Dataset to process"
)
parser.add_argument(
    "--version",
    type=str,
    default="v1",
    help="Version of the model used to generate data",
)
parser.add_argument(
    "--ckpt_name",
    type=str,
    default="grad_5000",
    help="Checkpoint name for the model used to generate data",
)

if __name__ == "__main__":
    args = parser.parse_args()
    main_data_dir = Path(args.main_data_dir)
    dataset = args.dataset
    version = args.version
    ckpt_name = args.ckpt_name
    speakers = dataset_2_speakers[dataset]

    if dataset != "MNGU0":
        processed_data_dir = main_data_dir / dataset / "processed_data"
        spkmetadata_filename = dataset_2_spkmetadata[dataset]
        linearmodel_name = dataset_2_linear_model[dataset]

    for speaker in speakers:
        print(
            f"Processing dataset: {dataset}, speaker: {speaker}, version: {version}, checkpoint: {ckpt_name}"
        )

        if dataset != "MNGU0":
            summary_df = get_sparc_summary_df(
                dataset, speaker, processed_data_dir, spkmetadata_filename
            )
            linearmodel_dir = (
                main_data_dir / dataset / "processed_data" / speaker / "linear_models"
            )
            linearmodel = joblib.load(linearmodel_dir / linearmodel_name)
            gt_ema_dir = main_data_dir / dataset / "src_data" / speaker
        else:
            src_data_dir = (
                main_data_dir / dataset / "src_data" / speaker / "ema_basic_data"
            )
            sparc_pred_dir = (
                main_data_dir
                / dataset
                / "arttts"
                / speaker
                / "encoded_audio_en"
                / "emasrc"
            )
            summary_df = get_sparc_summary_df_mngu0(src_data_dir, sparc_pred_dir)
            gt_ema_dir = (
                main_data_dir / dataset / "src_data" / speaker / "ema_basic_data"
            )

        # get the predicted EMAs from ART-TTS and SPARC
        # and add them to the previous table
        arttts_pred_dir = (
            main_data_dir
            / dataset
            / "arttts"
            / speaker
            / "arttts_pred"
            / version
            / ckpt_name
        )
        sparc_pred_dir = (
            main_data_dir / dataset / "arttts" / speaker / "encoded_audio_en" / "emasrc"
        )
        analysis_dir = main_data_dir / dataset / "arttts" / speaker / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        data = []
        for i, row in tqdm(summary_df.iterrows()):
            # get the same first columns as in the summary_df
            new_row = row.copy()
            new_row["pcc_no_dtw"] = row.pop("pcc")
            # del new_row["pcc"] forgot it but let's keep it for now for harmonization

            filestem = row["filestem"]
            arttts_ema = np.load(arttts_pred_dir / f"{filestem}.npy")
            enc_ema = arttts_ema[:12, :].T
            dec_ema = arttts_ema[14:26, :].T
            valid_enc = True
            valid_dec = True
            if np.isnan(enc_ema).any():
                print(f"{filestem} has nan in enc_ema")
                valid_enc = False
                continue
            if np.isnan(dec_ema).any():
                print(f"{filestem} has nan in dec_ema")
                valid_dec = False
                continue
            sparc_ema = np.load(sparc_pred_dir / f"{filestem}.npy")[:, :12]
            gt_ema = get_50Hz_ema(dataset, gt_ema_dir / f"{filestem}.ema")
            gt_ema = (gt_ema - np.mean(gt_ema, axis=0)) / np.std(gt_ema, axis=0)
            if dataset != "MNGU0":  # need to convert to speaker space
                if valid_enc:
                    enc_ema = (enc_ema - np.mean(enc_ema, axis=0)) / np.std(
                        enc_ema, axis=0
                    )
                else:
                    enc_ema = np.nan_to_num(enc_ema, nan=0.0, posinf=0.0, neginf=0.0)
                if valid_dec:
                    dec_ema = (dec_ema - np.mean(dec_ema, axis=0)) / np.std(
                        dec_ema, axis=0
                    )
                else:
                    dec_ema = np.nan_to_num(dec_ema, nan=0.0, posinf=0.0, neginf=0.0)
                sparc_ema = (sparc_ema - np.mean(sparc_ema, axis=0)) / np.std(
                    sparc_ema, axis=0
                )
                enc_ema = linearmodel.predict(enc_ema)
                dec_ema = linearmodel.predict(dec_ema)
                sparc_ema = linearmodel.predict(sparc_ema)

            dist_gt_enc, y_gt_enc_ada, y_enc_14_ada = normalized_dtw_score(
                gt_ema, enc_ema
            )
            dist_gt_dec, y_gt_dec_ada, y_dec_14_ada = normalized_dtw_score(
                gt_ema, dec_ema
            )
            dist_gt_sparc, y_gt_sparc_ada, y_sparc_14_ada = normalized_dtw_score(
                gt_ema, sparc_ema
            )
            dist_sparc_enc, y_sparc_enc_ada, y_enc_sparc_14_ada = normalized_dtw_score(
                sparc_ema, enc_ema
            )
            dist_sparc_dec, y_sparc_dec_ada, y_dec_sparc_14_ada = normalized_dtw_score(
                sparc_ema, dec_ema
            )

            pearson_enc, _ = pearsonr(y_gt_enc_ada, y_enc_14_ada)
            pearson_dec, _ = pearsonr(y_gt_dec_ada, y_dec_14_ada)
            pearson_sparc, _ = pearsonr(y_gt_sparc_ada, y_sparc_14_ada)
            pearson_enc_sparc, _ = pearsonr(y_sparc_enc_ada, y_enc_sparc_14_ada)
            pearson_dec_sparc, _ = pearsonr(y_sparc_dec_ada, y_dec_sparc_14_ada)

            new_row["dtw_gt_enc"] = dist_gt_enc
            new_row["dtw_gt_dec"] = dist_gt_dec
            new_row["dtw_gt_sparc"] = dist_gt_sparc
            new_row["dtw_sparc_enc"] = dist_sparc_enc
            new_row["dtw_sparc_dec"] = dist_sparc_dec
            new_row["pcc_gt_enc"] = np.mean(pearson_enc[:12])
            new_row["pcc_gt_dec"] = np.mean(pearson_dec[:12])
            new_row["pcc_gt_sparc"] = np.mean(pearson_sparc[:12])
            new_row["pcc_sparc_enc"] = np.mean(pearson_enc_sparc[:12])
            new_row["pcc_sparc_dec"] = np.mean(pearson_dec_sparc[:12])
            new_row["pred_rel_gap"] = np.abs(len(enc_ema) - len(gt_ema)) / len(gt_ema)
            new_row["enc_dtw_distortion"] = np.abs(
                len(y_enc_14_ada) - len(enc_ema)
            ) / len(enc_ema)
            new_row["dec_dtw_distortion"] = np.abs(
                len(y_dec_14_ada) - len(dec_ema)
            ) / len(dec_ema)
            new_row["sparc_dtw_distortion"] = np.abs(
                len(y_sparc_14_ada) - len(sparc_ema)
            ) / len(sparc_ema)
            new_row["valid_enc"] = valid_enc
            new_row["valid_dec"] = valid_dec
            data.append(new_row)

        res_df = pd.DataFrame(data)
        res_df.to_csv(
            analysis_dir / f"quanti_art_comp_{version}_{ckpt_name}.csv", index=False
        )
