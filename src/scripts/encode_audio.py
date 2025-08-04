from sparc import load_model
import argparse
from tqdm import tqdm
import logging
from pathlib import Path
import numpy as np
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--wav_dir",
    type=str,
)
parser.add_argument(
    "--save_dir",
    type=str,
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="ckpt/sparc_en.ckpt",  # or "ckpt/sparc_multi.ckpt"
)

"""
Do it as array jobs instead of multiprocessing
"""
# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # keeps progress bar clean
            self.flush()
        except Exception:
            self.handleError(record)


handler = TqdmLoggingHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def process_files(task_id, args, task_wavfiles):
    device = f"{args.device}"
    ckpt = args.ckpt_path
    save_dir = Path(args.save_dir)
    spk_emb_save_dir = save_dir / "spk_emb"
    ft_save_dir = save_dir / "emasrc"
    spk_emb_save_dir.mkdir(exist_ok=True, parents=True)
    ft_save_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Loading model for task_id %d", task_id)
    coder = load_model(ckpt=ckpt, device=device)
    logger.info("Model loaded for task_id %d", task_id)

    logger.info(f"Task_id {task_id} gpu processing {len(task_wavfiles)} files.")
    for wav_file in tqdm(
        task_wavfiles, desc=f"GPU {task_id}", position=1, dynamic_ncols=True
    ):
        save_name = str(wav_file).replace(str(args.wav_dir), "")
        save_name = Path(save_name).stem + ".npy"
        ft_save_path = ft_save_dir / save_name
        spk_emb_save_path = spk_emb_save_dir / save_name

        # if spk_emb_save_path.exists():
        #    continue

        try:
            with torch.inference_mode():
                outputs = coder.encode(wav_file, concat=True)
            np.save(ft_save_path, outputs["features"])
            np.save(spk_emb_save_path, outputs["spk_emb"])
        except Exception as e:
            logger.error(f"Error processing {wav_file}: {e}")
            continue

    logger.info(f"GPU {task_id} finished processing {len(task_wavfiles)} files.")


if __name__ == "__main__":
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    nb_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    args = parser.parse_args()
    logger.info(f"Starting the script for task {task_id}...")
    wav_dir = Path(args.wav_dir)

    wav_files = sorted([f for f in wav_dir.glob("**/*.wav")])
    logger.info(f"Found {len(wav_files)} audio files in {wav_dir}")

    # split the work into nb_tasks subtasks and process the task_id one
    wav_files_split = np.array_split(wav_files, nb_tasks)
    task_wavfiles = wav_files_split[task_id]
    process_files(task_id, args, task_wavfiles)
