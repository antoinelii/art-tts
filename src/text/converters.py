import re
import panphon
import numpy as np

from typing import Optional, List
from text.cmudict import CMUDict
from text import _clean_text
from text.symbols import _punctuation


_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")
_composed_re = re.compile(r"\b[a-zA-Z]+(?:-[a-zA-Z]+)+\b")  # composed words with dashes

_punctuation_list = list(_punctuation) + ["--"]

ft = panphon.FeatureTable()

# List of IPA ternary features
traits_list = [
    "syl",
    "son",
    "cons",
    "cont",
    "delrel",
    "lat",
    "nas",
    "strid",
    "voi",
    "sg",
    "cg",
    "ant",
    "cor",
    "distr",
    "lab",
    "hi",
    "lo",
    "back",
    "round",
    "velaric",
    "tense",
    "long",
    "hitone",
    "hireg",
]
N_traits = len(traits_list)
emb_dim = N_traits + 1  # add a dim for one hot space token
space_tok = np.zeros((1, emb_dim))
space_tok[0, -1] = 1


# CMU ARPabet to ipa conversion table from
# https://github.com/NVIDIA/NeMo/blob/main/scripts/tts_dataset_files/cmudict-arpabet_to_ipa_nv22.08.tsv
arpabet2ipa = {
    "AA": "ɑ",
    "AE": "æ",
    "AH0": "ə",
    "AH1": "ʌ",
    "AH2": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɜ˞",  #'ɝ' in the original
    "ER0": "ə˞",  #'ɚ' in the original
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}


def text_to_ipa(
    text: str,
    dictionary: Optional[CMUDict] = None,
    cleaner_names: List[str] = ["english_cleaners_v2"],
    remove_punctuation: bool = False,
) -> str:
    """
    Convert text to IPA characters sequence.
    """
    arp_list = text_to_arpabet(text, dictionary, cleaner_names)
    arp_list = check_arpabet(arp_list, remove_punctuation=remove_punctuation)
    if arp_list is None:
        print(f"Unable to convert to ARPAbet : {text}")
        return None
    else:
        # convert ARPAbet to IPA
        ipawords_list = [get_ipa_from_arp(w) for w in arp_list]
        return ipawords_list


def ipa_to_ternary(
    ipawords_list: List[str],
) -> np.ndarray:
    """
    Convert a list of IPA words to a ternary sequence.
    The sequence is a numpy array of shape (n_chars, N_traits)
    where n_chars is the number of characters in the IPA words
    and N_traits is the number of traits.
    The sequence is padded with zeros for punctuation and space tokens.
    """
    ternary_seq = []
    for word_ipa in ipawords_list:
        if ft.validate_word(word_ipa):
            emb_arr = ft.word_array(traits_list, word_ipa)  # shape: (n_chars, N_traits)
            ternary_seq.append(
                np.pad(emb_arr, ((0, 0), (0, 1)), mode="constant", constant_values=0)
            )
        elif word_ipa in _punctuation_list:
            ternary_seq.append(space_tok)
        else:
            print(f"Word not found in panphon: {word_ipa}")
            continue
    return np.concatenate(ternary_seq, axis=0)


############################
## text to ARPAbet functions
############################


def get_arpabet_dash(
    word: str,
    dictionary: Optional[CMUDict] = None,
) -> List[str]:
    """
    Get ARPAbet transcription for a word, handling dashed composed words.
    More specifically, if the word contains a dash, and is not in the dictionary,
    split the word at the dash and get ARPAbet for each part.
    The parts are then joined with a space
    """
    word_arpabet = dictionary.lookup(word)
    if word_arpabet is not None:
        return ["{" + word_arpabet[0] + "}"]
    elif _composed_re.match(word):
        words = word.split("-")
        words_arpabet = [get_arpabet_dash(w, dictionary)[0] for w in words]
        return words_arpabet
    else:
        return [word]


def text_to_arpabet(
    text: str,
    dictionary: Optional[CMUDict] = None,
    cleaner_names: List[str] = ["english_cleaners_v2"],
):
    """
    Convert text to ARPAbet words list.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
        text: input text
        dictionary: CMU dictionary
        cleaner_names: list of cleaner names
    Returns:
        ARPAbet words list (List[{"ARP1 ARP2 ...ARPN}" or "PUNC"])
    """
    arp_words = []
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            clean_text = _clean_text(text, cleaner_names)
            for w in clean_text.split(" "):
                arp_words += get_arpabet_dash(w, dictionary)
            break
        else:
            arp_words += text_to_arpabet(m.group(1), dictionary, cleaner_names)
            arp_words += ["{" + m.group(2) + "}"]
            text = m.group(3)
    return arp_words


def check_arpabet(
    arp_words: List[str],
    remove_punctuation: bool = False,
) -> List[str] | None:
    """
    Check if all words are ARPabet encoded (or punctuation).
    If not, return None.

    Args:
        arp_words: list of words
        remove_punctuation: if True, remove punctuation from the list
    Returns:
        list of words ("{ARP1 ARP2 ...ARPN}", or "PUNC") if all are valid
        None otherwise
    """
    mask_arp = [elem.startswith("{") and elem.endswith("}") for elem in arp_words]
    mask_punc = [elem in _punctuation_list for elem in arp_words]
    mask_invalid = [not (arp or punct) for arp, punct in zip(mask_arp, mask_punc)]
    if any(mask_invalid):
        return None
    elif remove_punctuation:
        return [elem for elem in arp_words if elem not in _punctuation_list]
    else:
        return arp_words


###########################
#### ARPAbet to IPA functions
###########################


def get_ipa_from_arp(arp_seq: str) -> str | None:
    """
    Get IPA transcription for an ARPabet sequence (format "{ARP1 ARP2 ...ARPN}").
    Handles punctuation words as well (".", ",", ...) by returning them as is.
    SHOULD BE CALLED AFTER check_arpabet() to ensure the ARPAbet sequence is valid.
    If the ARPAbet sequence is not valid, return None.

    Args:
        arp_seq: ARPAbet sequence or punctuation string
    Returns:
        IPA transcription : str or None if not found
                            ex : "pɹɪntɪŋ"
    """

    def arpchar_to_ipa(arp: str) -> str | None:
        """
        Get IPA transcription for an ARPAbet character.
        Try to find the original ARPAbet character. If not found,
        fallback to the ARPAbet character without stress markers
        """
        if arp in arpabet2ipa:
            return arpabet2ipa[arp]
        else:
            arp = arp.replace("1", "").replace("2", "").replace("0", "")
            return arpabet2ipa[arp]

    if arp_seq.startswith("{") and arp_seq.endswith("}"):
        arp_seq = arp_seq[1:-1].split(" ")
        ipa_seq = [arpchar_to_ipa(arp) for arp in arp_seq]
        return "".join(ipa_seq)
    elif arp_seq in _punctuation_list:
        return arp_seq
    else:
        print("Invalid ARPAbet sequence, should be checked with check_arpabet()")
        return None
