import numpy as np
import logging

from configs import params_v0

# from text.symbols import symbols
from pathlib import Path
from utils import parse_filelist

mylogger = logging.getLogger(__name__)
mylogger.setLevel(logging.INFO)

if __name__ == "__main__":
    mylogger.info("Calculating pitch statistics...")

    artic_dir = params_v0.artic_dir
    train_filelist_path = params_v0.train_filelist_path

    filepaths_and_text = parse_filelist(train_filelist_path)

    pitch_means = []
    pitch_stds = []
    for i, e in enumerate(filepaths_and_text):
        if i % 10 == 0:
            mylogger.info(f"Processing {i}th file...")
        filepath = e[0]
        art_filename = f"{Path(filepath).stem}.npy"
        preprocessed_fp = Path(artic_dir) / "emasrc" / art_filename
        pitch_arr = np.load(preprocessed_fp)[:, 12]
        pitch_mean = np.mean(pitch_arr)
        pitch_std = np.std(pitch_arr)
        pitch_means.append(pitch_mean)
        pitch_stds.append(pitch_std)

    # save pitch stats
    np.save(Path(artic_dir) / "../../../art-tts/pitch_means.npy", np.array(pitch_means))
    np.save(Path(artic_dir) / "../../../art-tts/pitch_stds.npy", np.array(pitch_stds))
