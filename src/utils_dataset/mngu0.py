import re
import numpy as np
import scipy

# from mngu0 ipa conversion table mngu0_s1_symbol_table.pdf
mngu02ipa = {
    "p": "p",
    "t": "t",
    "k": "k",
    "b": "b",
    "d": "d",
    "g": "ɡ",
    "m": "m",
    "n": "n",
    "N": "ŋ",
    "T": "θ",
    "D": "ð",
    "f": "f",
    "v": "v",
    "s": "s",
    "z": "z",
    "S": "ʃ",
    "Z": "ʒ",
    "tS": "t͡ʃ",  # "ʧ" in the original then "tʃ" but"t͡ʃ" to get a single embed
    "dZ": "d͡ʒ",  # "ʤ" in the original then "dʒ" but "d͡ʒ" to get a single embed
    "h": "h",
    "l": "l",
    "lw": "ɫ",
    "r": "ɹ",
    "j": "j",
    "w": "w",
    "m!": "m̩",
    "n!": "n̩",
    "l!": "l̩",
    "E": "ɛ",
    "a": "æ",
    "A": "ɑː",
    "@@": "ɜ",
    "@U": "əʊ",
    "Q": "ɒ",
    "O": "ɔː",
    "i": "iː",
    "I": "ɪ",
    "@": "ə",
    "V": "ʌ",
    "U": "ʊ",
    "u": "uː",
    "eI": "ɛɪ",
    "aI": "aɪ",
    "OI": "ɔɪ",
    "aU": "aʊ",
    "I@": "ɪə",
    "E@": "ɛə",
    "U@": "ʊə",
    "o^": "ɔ̃",
    "#": ".",  # silence equivalent to punctuation
}


def get_mngu0_sentence(utt_file: str) -> str:
    """
    Extracts the sentence from a MNGU0 .utt file.
    The sentence is found in the line starting with "Features" and is enclosed in 'iform' attribute.
    Args:
        utt_file (str): Path to the .utt file.
    Returns:
        str: The extracted sentence.
    """
    with open(utt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sentence = None
    for line in lines:
        # get the sentence
        if line.startswith("Features"):
            match = re.search(r'iform\s+"?(\\?"?[^";]+\\?"?)"?\s*;', line)
            if match:
                sentence = match.group(1)
                sentence = sentence.strip('"\\')  # clean it up
                break
    return sentence


def get_mngu0_phnm3(lab_file: str) -> np.ndarray:
    """
    Extracts phone information from a MNGU0 .lab file.
    Args:
        lab_file (str): Path to the .lab file.
    Returns:
        list: A list of tuples, each containing (start_time, end_time, phone)
              with phone converted to IPA char recognizable by panphon
              We call it a phnm3 object.
    """

    with open(lab_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    phones = []
    first_phone_idx = lines.index("#\n") + 1
    start_time = 0.0
    for line in lines[first_phone_idx:]:
        line = line.split()  # [endtime, '26', phone]
        end_time = float(line[0])
        phone = line[2]
        phones.append((start_time, end_time, mngu02ipa[phone]))
        start_time = end_time
    phones = np.array(phones, dtype=[("start", "f4"), ("end", "f4"), ("phone", "U10")])
    return phones


MNGU0_features = [
    "T3_py",
    "T3_pz",
    "T2_py",
    "T2_pz",
    "T1_py",
    "T1_pz",
    "jaw_py",
    "jaw_pz",
    "upperlip_py",
    "upperlip_pz",
    "lowerlip_py",
    "lowerlip_pz",
]


def read_mngu0_ema(raw_ema_fp):
    """Reads MNGU0 EMA data from a raw file and returns the relevant features.
    Args:
        raw_ema_fp (str or Path): Path to the raw EMA file.
    Returns:
        np.ndarray: A 2D numpy array containing the EMA data for the specified features.
    """
    columns = {}
    columns["time"] = 0
    columns["present"] = 1

    with open(raw_ema_fp, "rb") as f:
        dummy_line = f.readline()  # EST File Track
        datatype = f.readline().split()
        f.readline()  # Byte Order
        nframes = int(f.readline().split()[1])
        nchannels = int(f.readline().split()[1])
        dummy_line, datatype, nframes, nchannels

        while "CommentChar" not in str(f.readline(), "utf-8"):
            pass
        f.readline()  # empty line
        line = f.readline()

        while "EST_Header_End" not in str(line, "utf-8"):
            line = str(line, "utf-8").strip("\n")
            channel_number = int(line.split()[0].split("_")[1]) + 2
            channel_name = line.split()[1]
            columns[channel_name] = channel_number
            line = f.readline()

        ema_buffer = f.read()
        data = np.frombuffer(ema_buffer, dtype="float32")
        data_ = np.reshape(data, (-1, len(columns)))

        feat_idx = [columns[feat] for feat in MNGU0_features]

        ema_data = data_[:, feat_idx]
        ema_data = ema_data * 100  # initial data in  10^-5m , we turn it to mm
        nonan = 1
        if np.isnan(ema_data).sum() != 0:
            nonan = 0
            # Build a cubic spline out of non-NaN values.
            spline = scipy.interpolate.splrep(
                np.argwhere(~np.isnan(ema_data).ravel()),
                ema_data[~np.isnan(ema_data)],
                k=3,
            )
            # Interpolate missing values and replace them.
            for j in np.argwhere(np.isnan(ema_data)).ravel():
                ema_data[j] = scipy.interpolate.splev(j, spline)
    return ema_data, nonan
