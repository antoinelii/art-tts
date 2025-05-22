"""from https://github.com/keithito/tacotron"""

import re
from text import cleaners
from text.symbols import symbols
from text.cmudict import CMUDict
from typing import List, Optional
from text.cleaners import _punctuation_list

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")
_composed_re = re.compile(r"\b[a-zA-Z]+(?:-[a-zA-Z]+)+\b")  # composed words with dashes


def get_arpabet(word, dictionary):
    word_arpabet = dictionary.lookup(word)
    if word_arpabet is not None:
        return "{" + word_arpabet[0] + "}"
    else:
        return word


def text_to_sequence(text, cleaner_names=["english_cleaners"], dictionary=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary

    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    space = _symbols_to_sequence(" ")
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            clean_text = _clean_text(text, cleaner_names)
            if dictionary is not None:
                clean_text = [get_arpabet(w, dictionary) for w in clean_text.split(" ")]
                for i in range(len(clean_text)):
                    t = clean_text[i]
                    if t.startswith("{"):
                        sequence += _arpabet_to_sequence(t[1:-1])
                    else:
                        sequence += _symbols_to_sequence(t)
                    sequence += space
            else:
                sequence += _symbols_to_sequence(clean_text)
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    # remove trailing space
    if dictionary is not None:
        sequence = sequence[:-1] if sequence[-1] == space[0] else sequence
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"


###############################
## art-tts specific functions
###############################


def get_arpabet_dash(
    word: str,
    dictionary: Optional[CMUDict] = None,
) -> str:
    """
    Get ARPAbet transcription for a word, handling dashed composed words.
    More specifically, if the word contains a dash, and is not in the dictionary,
    split the word at the dash and get ARPAbet for each part.
    The parts are then joined with a space
    """
    word_arpabet = dictionary.lookup(word)
    if word_arpabet is not None:
        return "{" + word_arpabet[0] + "}"
    elif _composed_re.match(word):
        words = word.split("-")
        words_arpabet = [get_arpabet_dash(w, dictionary) for w in words]
        return " ".join(words_arpabet)
    else:
        return word


def text_to_arpabet(
    text: str,
    dictionary: Optional[CMUDict] = None,
    cleaner_names: List[str] = ["english_cleaners_v2"],
):
    """
    Convert text to ARPAbet sequence.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
        text: input text
        dictionary: CMU dictionary
        cleaner_names: list of cleaner names
    Returns:
        ARPAbet sequence
    """
    sequence = []
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            clean_text = _clean_text(text, cleaner_names)
            clean_text = [
                get_arpabet_dash(w, dictionary) for w in clean_text.split(" ")
            ]
            sequence += clean_text
            break
        else:
            sequence += text_to_arpabet(m.group(1), dictionary, cleaner_names)
            sequence += ["{" + m.group(2) + "}"]
            text = m.group(3)
    return sequence


def check_arpabet(
    words: List[str],
    remove_punctuation: bool = True,
) -> List[str] | None:
    """
    Check if all words are ARPabet encoded (or punctuation).
    If not, return None.

    Args:
        words: list of words
        remove_punctuation: if True, remove punctuation from the list
    Returns:
        list of words if all are valid, None otherwise
    """
    mask_arp = [elem.startswith("{") and elem.endswith("}") for elem in words]
    mask_punc = [elem in _punctuation_list for elem in words]
    mask_invalid = [not (arp or punct) for arp, punct in zip(mask_arp, mask_punc)]
    if any(mask_invalid):
        return None
    elif remove_punctuation:
        return [elem for elem in words if elem not in _punctuation_list]
    else:
        return words
