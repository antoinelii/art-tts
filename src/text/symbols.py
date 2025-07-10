"""from https://github.com/keithito/tacotron"""

from text import cmudict

_pad = "_"
_punctuation = "!'(),.:;? \"|"  # Added '"' and "|"
_punctuation_ori = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Prepend "@" to ARPAbet symbols to ensure uniqueness:
_arpabet = ["@" + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation_ori) + list(_letters) + _arpabet
