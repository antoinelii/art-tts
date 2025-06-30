from pathlib import Path

# Global constants for the project

# PATHS
SRC_DIR = Path(__file__).resolve().parent.parent

# MISC

##### MSPKA_EMA_ita #####
# EMA channels for the MSPKA dataset
# each file contains 21 channels: coordinate in 3D space
# (xul yul zul xll yll zll xui yui zui xli yli zli xtb ytb ztb xtm ytm ztm xtt ytt ztt)
# Midsagittal components that we want to use (cf Cheol paper)
# xul zul xll zll xli zli xtt ztt xtm ztm xtb ztb
# (See https://pure.rug.nl/ws/portalfiles/portal/199229973/labphon_6289_rebernik.pdf)
MSPKA_ema_idx_to_keep = [
    0,
    2,
    3,
    5,
    9,
    11,
    18,
    20,
    15,
    17,
    12,
    14,
]  # FOR THE MSPKA DATASET

##### PB2007 ############
# # EMA channels for the PB2007 dataset
# in angelo's dataset, ordered 01x,y ;02x,y ;03x,y ;04x,y ;06x,y ;05x,y
# i.e. lix, liy, ttx, tty, tdx, tdy, tbckx, tbcky, ulx, uly, llx, lly
# Do we really have the same markers than in SPARC? tback instead of tblade
# reorder the channels to match SPARC
pb2007_idx_to_keep = [8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5]

# splits Vowels, VCV, Monosyllabic words, Sentences
pb2007_splits = {
    "vowel": [
        (0, 18),
        (310, 325),
        (488, 489),
        (1086, 1087),
        (1088, 1089),
        (1090, 1091),
        (1092, 1093),
        (1094, 1095),
    ],
    "vcv": [(18, 310), (325, 488), (489, 599)],
    "mono": [
        (599, 992),
        (1079, 1080),
        (1083, 1084),
    ],
    "sentence": [
        (992, 1079),
        (1080, 1083),
        (1084, 1086),
        (1087, 1088),
        (1089, 1090),
        (1091, 1092),
        (1093, 1094),
        (1095, 1109),
    ],
}


def pb2007_id_type(splits):
    """
    Create a dictionary that maps the id of each file to its type (vowel, vcv, mono, sentence)
    and a dictionary that maps the type to the list of ids.
    """
    pb2007_id2type = {}
    pb2007_ids_per_type = {}
    for split_type, ranges in splits.items():
        ids_list = []
        for start, end in ranges:
            for i in range(start, end):
                pb2007_id2type[i] = split_type
            ids_list.extend(list(range(start, end)))
        pb2007_ids_per_type[split_type] = ids_list
    return pb2007_id2type, pb2007_ids_per_type


pb2007_id2type, pb2007_ids_per_type = pb2007_id_type(pb2007_splits)

##### mochatimit #####
# EMA channels for the mochatimit dataset

(
    "0 ui_x",
    "1 li_x",
    "2 ul_x",
    "3 ll_x",
    "4 tt_x",
    "5 ui_y",
    "6 li_y",
    "7 ul_y",
    "8 ll_y",
    "9 tt_y",
)
(
    "10 tb_x",
    "11 td_x",
    "12 ****",
    "13 v_x ",
    "14 bn_x",
    "15 tb_y",
    "16 td_y",
    "17 ****",
    "18 v_y ",
    "19 bn_y",
)
mochatimit_idx_to_keep = [2, 7, 3, 8, 1, 6, 4, 9, 10, 15, 11, 16]
