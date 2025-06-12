# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from model.utils import fix_len_compatibility

# data parameters
wavs_dir = "../../data/LJSpeech-1.1/wavs"
artic_dir = "../../data/LJSpeech-1.1/encoded_audio_en"
# wavs_dir = "../LJ_samples"
# artic_dir = "../LJ_samples/encoded_audio_en"
train_filelist_path = "resources/filelists/ljspeech/train_v0.txt"
valid_filelist_path = "resources/filelists/ljspeech/valid_v0.txt"
test_filelist_path = "resources/filelists/ljspeech/test_v0.txt"
cmudict_path = "resources/cmu_dictionary"
sparc_ckpt_path = "ckpt/sparc_en.ckpt"
add_blank = True
n_feats = 16  # 14 to 16, actually need n_feats // 2**2 fot compatibility with U-Net
n_spks = 1  # 247 for Libri-TTS filelist and 1 for LJSpeech
spk_emb_dim = 64
# SPARC features order
# "ULX", "ULY", "LLX", "LLY", "LIX", "LIY",
# "TTX", "TTY", "TBX", "TBY", "TDX", "TDY"
# "pitch", "loudness"
reorder_feats = [0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11, 15, 13]
pitch_idx = reorder_feats[12]  # pitch channel index among n_feats
non_pitch_idx = [
    i for i in range(n_feats) if i != pitch_idx
]  # non-pitch channels indices
plot_norm_pitch = False  # whether to plot normalized pitch or not
merge_diphtongues = True  # whether to merge diphthongs in IPA embedding

# encoder parameters
n_ipa_feats = 25
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 1  # normally 2 but need to divide n_ipa_feats
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = "logs/new_exp"
test_size = 4
n_epochs = 10000  # 10000 normally
batch_size = 16
learning_rate = 1e-4
seed = 37
save_every = 5
val_every = 5
patience = 3  # patience each save_every epochs
# out_size = fix_len_compatibility(2 * 22050 // 256)      # 2* sr/hop_size meaning 2 seconds of audio
out_size = fix_len_compatibility(2 * 50)  # 2* art_sr meaning 2 seconds of audio
