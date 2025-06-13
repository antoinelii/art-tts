import numpy as np

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
    "ng": "ŋ",
    "l": "l",
    "r": "ɹ",
    "j": "j",
    "w": "w",
    "dZ": "dʒ",
    "tS": "tʃ",
    "dz": "dz",
    "ts": "ts",
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
    "ddZ": "dʒː",
    "ddz": "dzː",
    "ttS": "tʃː",
    "tts": "tsː",
    "nf": "nf",
    "LL": "lː",
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
            phnm3.append((float(s), float(e), phone))
        elif len(line) == 3:
            s, e, phone = line
            phnm3.append((float(s), float(e), phone))
    phnm3 = [(s, e, mspka2ipa[phone]) for s, e, phone in phnm3]
    phnm3 = np.array(phnm3, dtype=[("start", "f4"), ("end", "f4"), ("phone", "U10")])
    return phnm3
