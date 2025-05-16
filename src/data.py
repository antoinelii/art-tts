
import random
import numpy as np

import torch
import torchaudio as ta

#from text import text_to_sequence, cmudict
#from text.symbols import symbols
#from utils import parse_filelist, intersperse
#from model.utils import fix_len_compatibility
#from params import seed as random_seed

import sys
sys.path.insert(0, 'hifi-gan')
from meldataset import mel_spectrogram