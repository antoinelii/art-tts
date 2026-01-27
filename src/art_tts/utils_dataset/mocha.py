import numpy as np
from utils_ema.cst import mochatimit_idx_to_keep


def get_mocha_sentence(trans_file: str) -> str:
    """
    Reads the first line of a Mocha transcription file and returns the sentence.
    """
    with open(trans_file, "r") as f:
        sentence = f.readline().strip()
    return sentence


def get_mocha_phnm3(phnm_file: str) -> np.ndarray:
    """
    Extracts phone information from a mocha .phnm file.
    Args:
        phnm_file (str): Path to the .phnm file.
    Returns:
        list: A list of tuples, each containing (start_time, end_time, phone)
              with phone converted to IPA char recognizable by panphon
              We call it a phnm3 object.
    """

    def norm_special(phone):
        if phone == "sil":
            return "."
        elif phone == "ɚ":
            return "ə˞"
        elif phone == "ɝ":
            return "ɜ˞"
        return phone

    with open(phnm_file, "r") as f:
        phnm3 = f.readlines()
    phnm3 = [line.strip().split() for line in phnm3 if line.strip()]
    phnm3 = [(float(s), float(e), norm_special(phone)) for s, e, phone in phnm3]
    phnm3 = np.array(phnm3, dtype=[("start", "f4"), ("end", "f4"), ("phone", "U10")])
    return phnm3


def _read_mocha_ema(src_ema_fp):
    with open(src_ema_fp, "rb") as f:
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
    assert data.size % num_features == 0, "Data does not align to expected frame size"
    frames = data.reshape(-1, num_features)
    # parse into dictionary
    parsed = {
        "time": frames[:, 0],
        "valid": frames[:, 1],
        "ema": frames[:, 2:22],
        "header": header_lines,
    }
    return parsed


def get_mochatimit_ema(src_ema_fp):
    ema = _read_mocha_ema(src_ema_fp)["ema"]  # shape (n_timesteps, 20)
    ema = ema[:, mochatimit_idx_to_keep]  # shape (n_timesteps, 12)
    ema = ema.astype(np.float32)
    return ema
