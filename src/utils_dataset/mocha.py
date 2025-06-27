import numpy as np


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
