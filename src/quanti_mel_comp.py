from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent))
sys.path.insert(0, "hifi-gan")
from meldataset import mel_spectrogram

import joblib
import pandas as pd
import numpy as np

from paths import DATA_DIR


from scipy.stats import pearsonr
from metrics import normalized_dtw_score

from utils_dataset.mngu0 import read_mngu0_ema
import torchaudio as ta

from tqdm import tqdm

import argparse

dataset_2_spkmetadata = {
    "MSPKA_EMA_ita": "mixed_speaker_metadata_100Hz.joblib",
    "pb2007": "1.0_speaker_metadata_100Hz.joblib",
    "mocha_timit": "mixed_speaker_metadata_100Hz.joblib",
}

dataset_2_speakers = {
    "MSPKA_EMA_ita": ["cnz", "lls", "olm"],
    "pb2007": ["spk1"],
    "mocha_timit": ["faet0", "ffes0", "fsew0", "maps0", "mjjn0", "msak0"],
    "MNGU0": ["s1"],
}

n_fft = 1024
n_mels = 80
sample_rate = 22050
hop_length = 256
win_length = 1024
f_min = 0.0
f_max = 8000


def get_mel(wav_fp, sample_rate=22050):
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
        n_fft,
        n_mels,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        center=False,
    ).squeeze()
    return mel


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
    "--dataset", type=str, default="MSPKA_EMA_ita", help="Dataset to process"
)
parser.add_argument(
    "--version",
    type=str,
    default="v2",
    help="Version of the model used to generate data",
)
parser.add_argument(
    "--ckpt_name",
    type=str,
    default="grad_2000",
    help="Checkpoint name for the model used to generate data",
)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset
    version = args.version
    ckpt_name = args.ckpt_name
    speakers = dataset_2_speakers[dataset]

    if dataset != "MNGU0":
        processed_data_dir = DATA_DIR / dataset / "processed_data"
        spkmetadata_filename = dataset_2_spkmetadata[dataset]

    for speaker in speakers:
        print(
            f"Processing dataset: {dataset}, speaker: {speaker}, version: {version}, checkpoint: {ckpt_name}"
        )

        if dataset != "MNGU0":
            summary_df = get_sparc_summary_df(
                dataset, speaker, processed_data_dir, spkmetadata_filename
            )
            gt_wav_dir = DATA_DIR / dataset / "src_data" / speaker
        else:
            src_data_dir = DATA_DIR / dataset / "src_data" / speaker / "ema_basic_data"
            sparc_pred_dir = (
                DATA_DIR / dataset / "arttts" / speaker / "encoded_audio_en" / "emasrc"
            )
            summary_df = get_sparc_summary_df_mngu0(src_data_dir, sparc_pred_dir)
            gt_wav_dir = DATA_DIR / dataset / "src_data" / speaker / "wav_16kHz"

        # get the predicted Mels from Grad-TTS and SPARC
        # and add them to the previous table
        hifigan_pred_dir = (
            DATA_DIR
            / dataset
            / "arttts"
            / speaker
            / "hifigan_pred"
            / version
            / ckpt_name
        )
        sparc_pred_dir = (
            DATA_DIR / dataset / "arttts" / speaker / "hifigan_pred" / "sparc"
        )
        analysis_dir = DATA_DIR / dataset / "arttts" / speaker / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        data = []
        for i, row in tqdm(summary_df.iterrows()):
            # get the same first columns as in the summary_df
            new_row = row.copy()
            new_row["pcc_no_dtw"] = row.pop("pcc")
            del new_row["pcc"]  # forgot it but let's keep it for now for harmonization

            filestem = row["filestem"]
            enc_mel = get_mel(hifigan_pred_dir / f"{filestem}_encoder.wav").T.numpy()[
                :, ::4
            ]  # for speed up
            dec_mel = get_mel(hifigan_pred_dir / f"{filestem}_decoder.wav").T.numpy()[
                :, ::4
            ]  # for speed up
            sparc_mel = get_mel(sparc_pred_dir / f"{filestem}.wav").T.numpy()[
                :, ::4
            ]  # for speed up
            gt_mel = get_mel(gt_wav_dir / f"{filestem}.wav").T.numpy()[
                :, ::4
            ]  # for speed up

            dist_gt_enc, y_gt_enc_ada, y_enc_14_ada = normalized_dtw_score(
                gt_mel, enc_mel
            )
            dist_gt_dec, y_gt_dec_ada, y_dec_14_ada = normalized_dtw_score(
                gt_mel, dec_mel
            )
            dist_gt_sparc, y_gt_sparc_ada, y_sparc_14_ada = normalized_dtw_score(
                gt_mel, sparc_mel
            )
            dist_sparc_enc, y_sparc_enc_ada, y_enc_sparc_14_ada = normalized_dtw_score(
                sparc_mel, enc_mel
            )
            dist_sparc_dec, y_sparc_dec_ada, y_dec_sparc_14_ada = normalized_dtw_score(
                sparc_mel, dec_mel
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
            new_row["pred_rel_gap"] = np.abs(len(enc_mel) - len(gt_mel)) / len(gt_mel)
            new_row["enc_dtw_distortion"] = np.abs(
                len(y_enc_14_ada) - len(enc_mel)
            ) / len(enc_mel)
            new_row["dec_dtw_distortion"] = np.abs(
                len(y_dec_14_ada) - len(dec_mel)
            ) / len(dec_mel)
            new_row["sparc_dtw_distortion"] = np.abs(
                len(y_sparc_14_ada) - len(sparc_mel)
            ) / len(sparc_mel)
            data.append(new_row)

        res_df = pd.DataFrame(data)
        res_df.to_csv(
            analysis_dir / f"quanti_art_comp_{version}_{ckpt_name}.csv", index=False
        )
