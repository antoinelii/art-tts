from sparc import load_model
import argparse
import tqdm
import sys
from pathlib import Path
import numpy as np
import time
import torch

parser = argparse.ArgumentParser()
# parser.add_argument("--rank",type=int,default=0)
# parser.add_argument("--n",type=int,default=1)
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
# parser.add_argument("--batch_size",type=int,default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    device = args.device
    ckpt = args.ckpt_path
    wav_dir = Path(args.wav_dir)
    save_dir = Path(args.save_dir)
    spk_emb_save_dir = save_dir / "spk_emb"
    spk_emb_save_dir.mkdir(exist_ok=True)
    ft_save_dir = save_dir / "emasrc"
    ft_save_dir.mkdir(exist_ok=True)
    coder = load_model(ckpt=ckpt, device=device)

    wav_files = [f for f in wav_dir.glob("**/*.flac")] + [
        f for f in wav_dir.glob("**/*.wav")
    ]

    print(f"Found {len(wav_files)} audio files in {wav_dir}", flush=True)
    start = time.time()
    print(f"starting time: {tqdm.tqdm.format_interval(start)}")
    ##create a data loader
    # from torch.utils.data import DataLoader
    ##create a dataset with the files found
    # from torch.utils.data import Dataset
    # class WavDataset(Dataset):
    #    def __init__(self, wav_files):
    #        self.wav_files = wav_files
    #    def __len__(self):
    #        return len(self.wav_files)
    #    def __getitem__(self, idx):
    #        return self.wav_files[idx]
    #
    # wav_files = WavDataset(wav_files)
    # data_loader = DataLoader(wav_files, batch_size=4, shuffle=False)

    for wav_file in tqdm.tqdm(
        wav_files, file=sys.stdout, dynamic_ncols=False, disable=False
    ):
        save_name = str(wav_file).replace(str(wav_dir), "")
        save_name = Path(save_name).stem + ".npy"
        path_depth = len(save_name.split("/"))

        ft_save_path = ft_save_dir / save_name
        spk_emb_save_path = spk_emb_save_dir / save_name

        if (spk_emb_save_path).exists():
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
            with torch.no_grad():
                outputs = coder.encode(wav_file, concat=True)
            # dictionary with keys: "features", "spk_emb"
            # "features" = 12 EMA + pitch + loudness + periodicity
            # "spk_emb" = 64 features embedding vector
        except FileNotFoundError:
            continue
        np.save(ft_save_path, outputs["features"])
        np.save(spk_emb_save_path, outputs["spk_emb"])

    print(f"Finished processing {len(wav_files)} audio files.")
    end = time.time()
    print(f"Execution time: {end - start:.4f} seconds")
