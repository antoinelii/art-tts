import importlib
import argparse
from tqdm import tqdm
import logging
from pathlib import Path
import numpy as np

dataset_params = {
    "mngu0": ("utils_dataset.mngu0", "get_mngu0_phnm3", ".lab"),
    "mspka": ("utils_dataset.mspka", "get_mspka_phnm3", ".lab"),
    "pb2007": ("utils_dataset.pb2007", "get_pb2007_phnm3", ".phone"),
    "mocha": ("utils_dataset.mocha", "get_mocha_phnm3", ".phnm"),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--phnm_dir",
    type=str,
)
parser.add_argument(
    "--save_dir",
    type=str,
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="mngu0",  # or "mspka", "pb2007", "mocha"
)

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    args = parser.parse_args()
    phnm_dir = Path(args.phnm_dir)
    save_dir = Path(args.save_dir)
    dataset_name = args.dataset_name

    if not phnm_dir.exists():
        raise FileNotFoundError(f"Phoneme directory {phnm_dir} does not exist.")

    save_dir.mkdir(parents=True, exist_ok=True)

    mod_name, func_name, ext = dataset_params[dataset_name]
    mod = importlib.import_module(mod_name)
    get_phnm3 = getattr(mod, func_name)

    phnm_files = sorted([f for f in phnm_dir.glob(f"*{ext}")])
    logger.info(f"Found {len(phnm_files)} audio files in {phnm_dir}")

    for phnm_file in tqdm(phnm_files, desc=f"Processing {dataset_name} phoneme files"):
        try:
            ipa_phnm3 = get_phnm3(phnm_file)
            # Save the phoneme data
            save_path = save_dir / f"{phnm_file.stem}_phnm3.npy"
            np.save(save_path, ipa_phnm3)
            logger.info(f"Saved phoneme data to {save_path}")
        except Exception as e:
            logger.error(f"Error processing {phnm_file}: {e}")
