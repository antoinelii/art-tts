# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from model.utils import fix_len_compatibility


# data parameters
data_root_dir = "../../data"  # for jean-zay relative to src
data_root_dir = "/lustre/fsn1/projects/rech/rec/commun/data"  # for jean-zay scratch
# data_root_dir = "../../../../scratch2/ali/data" #for oberon2 relative to src

train_filelist_path = "resources/filelists/ljspeech/train_v2.txt"
valid_filelist_path = "resources/filelists/ljspeech/valid_v2.txt"
test_filelist_path = "resources/filelists/ljspeech/test_v2.txt"
cmudict_path = "resources/cmu_dictionary"
add_blank = True  # keep it? unfair to test datasets
gradtts_text_conv = False  # Wether to use grad-tts text to symbol conversion (imperfect
# and lacunary) or mine

n_feats = 80
n_spks = 1  # 247 for Libri-TTS filelist and 1 for LJSpeech
spk_emb_dim = 64
n_fft = 1024
sample_rate = 22050
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = "logs/v2_phnmtext"
test_size = 4
n_epochs = 10000
batch_size = 16
learning_rate = 1e-4
random_seed = 37
val_every = 5
save_every = 5
patience = 10
out_size = fix_len_compatibility(2 * 22050 // 256)
