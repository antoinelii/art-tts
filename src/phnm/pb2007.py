import numpy as np

ema_sr = 100

# chatgpt generated mapping from set of pb2007 phones
pb20072ipa = {
    "__": ".",  # Longue pause
    "_": ".",  # Courte pause (liaison douce ou légère pause)
    # Voyelles orales
    "a": "a",
    "e^": "ɛ",
    "e": "e",
    "i": "i",
    "y": "y",
    "u": "u",
    "o^": "ɔ",
    "o": "o",
    "x": "ø",
    "x^": "œ",
    "q": "ə",  # Schwa
    # Voyelles nasales
    "a~": "ɑ̃",
    "e~": "ɛ̃",
    "x~": "œ̃",  # pourrait varier selon l'accent, parfois [ɛ̃]
    "o~": "ɔ̃",
    # Consonnes sourdes
    "p": "p",
    "t": "t",
    "k": "k",
    "f": "f",
    "s": "s",
    "s^": "ʃ",
    # Consonnes sonores
    "b": "b",
    "d": "d",
    "g": "ɡ",
    "v": "v",
    "z": "z",
    "z^": "ʒ",
    # Nasales, liquides, semi-voyelles
    "m": "m",
    "n": "n",
    "r": "ʁ",
    "l": "l",
    "w": "w",
    "h": "h",
    "j": "j",
}


def get_pb2007_phnm3(phone_file: str) -> np.ndarray:
    with open(phone_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    lines = [line.split(" ") for line in lines]

    phnm3 = []
    for line in lines:
        if len(line) == 3:
            s_frame, e_frame, phone = line
            s_sec = float(s_frame) / ema_sr
            e_sec = float(e_frame) / ema_sr
            phnm3.append((s_sec, e_sec, phone))
    phnm3 = [(s, e, pb20072ipa[phone]) for s, e, phone in phnm3]
    phnm3 = np.array(phnm3, dtype=[("start", "f4"), ("end", "f4"), ("phone", "U10")])
    return phnm3
