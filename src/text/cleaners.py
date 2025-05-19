""" from https://github.com/keithito/tacotron """

import re
from unidecode import unidecode
from .numbers import normalize_numbers
from .symbols import _punctuation

_whitespace_re = re.compile(r'\s+')

_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
    ('&', 'and'),       # Added '&'
]]

_punctuation_list = list(_punctuation) + ["--"]

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text

def english_cleaners_v2(text):
    """"
    This cleaner is used for the art-TTS model.

    Added the following features:
        - pad punctuation with left and right space (to isolate from words)
        - strip introduced head/tail extra spaces
    """
    def pad_punctuation(text):
        """
        Isolate punctuation in the text by adding spaces around punctuation characters.
        """
        processed_text = ''.join(f" {char} " if char in _punctuation_list else char for char in text)
        return processed_text
    text = lowercase(text)
    text = expand_numbers(text)
    text = convert_to_ascii(text)
    text = expand_abbreviations(text)
    text = pad_punctuation(text)
    text = collapse_whitespace(text)
    text = text.strip()
    return text