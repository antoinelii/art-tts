from sparc import load_model
import argparse
from tqdm import tqdm
import logging
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp

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
parser.add_argument("--ckpt_path", type=str, default="ckpt/sparc_en.ckpt")
parser.add_argument("--num_gpus", type=int, default=4)

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


def process_files(rank, args, wav_files_split):
    device = f"{args.device}:{rank}"
    ckpt = args.ckpt_path
    save_dir = Path(args.save_dir)
    spk_emb_save_dir = save_dir / "spk_emb"
    ft_save_dir = save_dir / "emasrc"
    logger.info("Loading model on GPU %d", rank)
    coder = load_model(ckpt=ckpt, device=device)
    logger.info("Model loaded on GPU %d", rank)

    gpu_wav_files = wav_files_split[rank]
    logger.info(f"GPU {rank} processing {len(gpu_wav_files)} files.")
    for wav_file in tqdm(
        gpu_wav_files, desc=f"GPU {rank}", position=rank, dynamic_ncols=True
    ):
        save_name = str(wav_file).replace(str(args.wav_dir), "")
        save_name = Path(save_name).stem + ".npy"
        ft_save_path = ft_save_dir / save_name
        spk_emb_save_path = spk_emb_save_dir / save_name

        if spk_emb_save_path.exists():
            continue

        def _recursive_path_solver(file_path):
            if file_path.exists():
                return
            elif file_path.parent.exists():
                file_path.mkdir(exist_ok=True)
                return
            else:
                _recursive_path_solver(file_path.parent)

        _recursive_path_solver(spk_emb_save_path.parent)
        _recursive_path_solver(ft_save_path.parent)

        try:
            with torch.inference_mode():
                outputs = coder.encode(wav_file, concat=True)
            np.save(ft_save_path, outputs["features"])
            np.save(spk_emb_save_path, outputs["spk_emb"])
        except Exception as e:
            logger.error(f"Error processing {wav_file}: {e}")
            continue

    logger.info(f"GPU {rank} finished processing {len(gpu_wav_files)} files.")


if __name__ == "__main__":
    args = parser.parse_args()
    logger.info("Starting the script...")
    wav_dir = Path(args.wav_dir)
    save_dir = Path(args.save_dir)
    spk_emb_save_dir = save_dir / "spk_emb"
    spk_emb_save_dir.mkdir(exist_ok=True)
    ft_save_dir = save_dir / "emasrc"
    ft_save_dir.mkdir(exist_ok=True)

    wav_files = [f for f in wav_dir.glob("**/*.flac")] + [
        f for f in wav_dir.glob("**/*.wav")
    ]
    logger.info(f"Found {len(wav_files)} audio files in {wav_dir}")

    # Split files across GPUs
    num_gpus = args.num_gpus
    wav_files_split = np.array_split(wav_files[:400], num_gpus)
    # Start multiprocessing
    logger.info(f"Starting processing on {num_gpus} GPUs...")
    mp.spawn(process_files, args=(args, wav_files_split), nprocs=num_gpus, join=True)

    logger.info("Finished processing all audio files.")
