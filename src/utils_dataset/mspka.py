import numpy as np
from utils_ema.cst import MSPKA_ema_idx_to_keep

# used chatgpt then https://www.leskoff.com/s01763-0 to get a more
# italian specific version of the phones
mspka2ipa = {
    "a": "a",
    "e": "e",
    "E1": "ɛ",
    "i": "i",
    "o": "o",
    "O1": "ɔ",
    "u": "u",
    "b": "b",
    "d": "d",
    "g": "ɡ",
    "p": "p",
    "t": "t",
    "k": "k",
    "f": "f",
    "v": "v",
    "s": "s",
    "z": "z",
    "SS": "ʃ",
    "JJ": "ʒ",
    "m": "m",
    "n": "n",
    "ng": "ɲ",  # "ŋ" but "ɲ" more italian
    "l": "l",
    "r": "ɾ",  # "ɹ" but "ɾ" more italian r
    "j": "j",
    "w": "w",
    "dZ": "d͡ʒ",  # "dʒ" but "d͡ʒ" to get a single embed
    "tS": "t͡ʃ",  # "tʃ" but "t͡ʃ" to get a single embed
    "dz": "d͡z",  # "dz",but "d͡z" to get a single embed
    "ts": "t͡s",  # "ts" but "t͡s" to get a single embed
    "dd": "dː",
    "tt": "tː",
    "ss": "sː",
    "pp": "pː",
    "kk": "kː",
    "ll": "lː",
    "rr": "rː",
    "nn": "nː",
    "mm": "mː",
    "gg": "ɡː",
    "vv": "vː",
    "ddZ": "d͡ʒː",  # "dʒ" but "d͡ʒ" to get a single embed
    "ddz": "d͡zː",  # "dzː" but "d͡zː" to get a single embed
    "ttS": "t͡ʃː",  # "tʃ" but "t͡ʃ" to get a single embed
    "tts": "t͡sː",  # "tsː" but "t͡sː" to get a single embed
    "nf": "nf",
    "LL": "ʎ",  # "lː" but "ʎ" to get the gli
    "bb": "bː",
    "ff": "fː",
    "sil": ".",
}


def get_mspka_sentence(lab_file):
    """ """
    with open(lab_file, "rb") as f:
        raw = f.read()
    # Need more work to decode chars like 'ì'
    # Step 1: Interpret the escaped octal sequences as real characters
    as_str = raw.decode("latin1")  # preserve bytes
    decoded_bytes = as_str.encode("latin1").decode("unicode_escape").encode("latin1")
    # Step 2: Decode the real UTF-8 bytes
    decoded_text = decoded_bytes.decode("utf-8")
    lines = decoded_text.splitlines()
    lines = [line.strip().split(" ") for line in lines if line.strip()]

    phnm3 = []
    sentence = []
    for line in lines:
        if len(line) == 4:  # beginning of word
            s, e, phone, word = line
            if line[2] != "sil":  # real word not silence
                sentence.append(word)
    return " ".join(sentence), phnm3


def get_mspka_phnm3(lab_file):
    """ """
    with open(lab_file, "rb") as f:
        raw = f.read()
    # Need more work to decode chars like 'ì'
    # Step 1: Interpret the escaped octal sequences as real characters
    as_str = raw.decode("latin1")  # preserve bytes
    decoded_bytes = as_str.encode("latin1").decode("unicode_escape").encode("latin1")
    # Step 2: Decode the real UTF-8 bytes
    decoded_text = decoded_bytes.decode("utf-8")
    lines = decoded_text.splitlines()
    lines = [line.strip().split(" ") for line in lines if line.strip()]

    phnm3 = []
    for line in lines:
        if len(line) == 4:  # beginning of word
            s, e, phone, word = line
        elif len(line) == 3:
            s, e, phone = line
        if phone != "nf":
            phnm3.append((float(s), float(e), phone))
        else:  # if phone is "nf", we split it into "n" and "f"
            delta = float(e) - float(s)
            delta_n = delta / 2
            phnm3.append((float(s), float(s) + delta_n, "n"))
            phnm3.append((float(s) + delta_n, float(e), "f"))
    phnm3 = [(s, e, mspka2ipa[phone]) for s, e, phone in phnm3]
    phnm3 = np.array(phnm3, dtype=[("start", "f4"), ("end", "f4"), ("phone", "U10")])
    return phnm3


def get_MSPKA_ema(src_ema_fp):
    with open(src_ema_fp, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]

    ema = np.array(lines, dtype=np.float32)  # shape (n_channels, n_timesteps)
    ema = ema[MSPKA_ema_idx_to_keep, :].T  # shape (n_timesteps, n_channels)
    return ema
