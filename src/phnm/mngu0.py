import re
import numpy as np

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
    "tS": "tʃ",  #'ʧ' in the original
    "dZ": "dʒ",  # 'ʤ' in the original
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
