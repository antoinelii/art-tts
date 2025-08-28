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

sparc_ckpt_path = "ckpt/sparc_multi.ckpt"
# add_blank = True

n_feats = 16  # 14 to 16, actually need n_feats // 2**2 fot compatibility with U-Net
n_spks = 1  # 247 for Libri-TTS filelist and 1 for LJSpeech
spk_emb_dim = 64
# SPARC features order (according to paper prob false)
# "ULX", "ULY", "LLX", "LLY", "LIX", "LIY",
# "TTX", "TTY", "TBX", "TBY", "TDX", "TDY"
# "pitch", "loudness"
reorder_feats = [0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11, 15, 13]
pitch_idx = reorder_feats[12]  # pitch channel index among n_feats
log_normalize_loudness = False
loudness_idx = reorder_feats[13]  # loudness channel index among n_feats
non_pitch_idx = [
    i for i in range(n_feats) if i != pitch_idx
]  # non-pitch channels indices
plot_norm_pitch = False  # whether to plot normalized pitch or not
merge_diphtongues = False  # whether to merge diphthongs in IPA embedding

# encoder parameters
n_ipa_feats = 26  # 25 in v1, here we add phoneme repetition count at 26th dimension
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2  # 1 in v1 but 2 in Grad-TTS, here we can use 2 since n_ipa_feats is even
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = "logs/msml1h"
test_size = 8
n_epochs = 5000  # 10000 normally
batch_size = 16
learning_rate = 1e-4
random_seed = 37
save_every = 50
val_every = 50
# out_size = fix_len_compatibility(2 * 22050 // 256)      # 2* sr/hop_size meaning 2 seconds of audio
out_size = fix_len_compatibility(2 * 50)  # 2* art_sr meaning 2 seconds of audio

# train data parameters
dataset_dir = f"{data_root_dir}/VoxCommunis"
suffix = "-1h"  # "-1h" or "-20h"

train_manifest = f"{dataset_dir}/train{suffix}/manifests"
train_alignment = f"{dataset_dir}/train{suffix}/alignments"

val_manifest = f"{dataset_dir}/dev{suffix}/manifests"
val_alignment = f"{dataset_dir}/dev{suffix}/alignments"

test_manifest = f"{dataset_dir}/test{suffix}/manifests"
test_alignment = f"{dataset_dir}/test{suffix}/alignments"

separate_files = True  # whether to use a directory of manifest files or a manifest file
if not separate_files:  # monolingual
    lang = "it"  # choose language here
    train_manifest += f"/{lang}.tsv"
    val_manifest += f"/{lang}.tsv"
    test_manifest += f"/{lang}.tsv"
    train_alignment += f"/{lang}.align"
    val_alignment += f"/{lang}.align"
    test_alignment += f"/{lang}.align"

lang_codes = [
    "ka",
    "ja",
    "ba",
    "ro",
    "hi",
    "uz",
    "tt",
    "el",
    "sr",
    "mt",
    "yo",
    "be",
    "uk",
    "hy-AM",
    "sk",
    "ckb",
    "ur",
    "tr",
    "vi",
    "sq",
    "bg",
    "ta",
    "sv-SE",
    "eu",
    "id",
    "sw",
    "tk",
    "kmr",
    "dv",
    "ha",
    "zh-HK",
    "bn",
    "mn",
    "zh-CN",
    "yue",
    "lij",
    "fr",
    "hsb",
    "cv",
    "ko",
    "nl",
    "ug",
    "mr",
    "ab",
    "it",
    "lt",
    "sl",
    "kk",
    "pa-IN",
    "ru",
    "cs",
    "gn",
    "ml",
    "nan-tw",
    "th",
    "pt",
    "ky",
    "pl",
    "ca",
    "myv",
    "hu",
    "rw",
    "am",
]

insufficient_langs = ["kk", "am", "ur", "sq"]
zeroshot_langs = ["eu", "ka", "ab", "gn", "sw", "ha", "ko", "myv"]
exclude_langs = insufficient_langs + zeroshot_langs
# 0.9 pretty much all the languages have the same amount of time
# just not the same amount of samples. We still boost the languages
# with less samples but not too much.
upsample_factor = 0.9
