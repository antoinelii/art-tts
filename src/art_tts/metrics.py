import numpy as np
from tslearn.metrics import dtw_path

# Perform DTW between y_gt and y_enc_14
# use example
# path, distance = dtw_path(s1, s2)
# path list of integer pairs


def signals_from_path(
    s1: np.ndarray, s2: np.ndarray, path_s1_s2: list[tuple[int, int]]
) -> np.ndarray:
    """
    Adapt s1 and s2 to match the alignment provided by path
    calculated using DTW.

    Parameters:
    - s1: array to adapt(shape: [n_frames_1, n_features])
    - s2: array to adapt (shape: [n_frames_2, n_features])
    - path: list of tuples (i, j) representing the DTW path
            i being the index in s1 and j in s2. path is obtained from
            dtw_path(s1, s2) to adapt s2 to s1 length or dtw_path_from_metric

    Returns:
    - s1_adapted: Adapted array
    - s2_adapted: Adapted array
    """
    s1_adapted = np.zeros((len(path_s1_s2), s1.shape[1]), dtype=s1.dtype)
    s2_adapted = np.zeros((len(path_s1_s2), s2.shape[1]), dtype=s2.dtype)
    for i, (i1, i2) in enumerate(path_s1_s2):
        s1_adapted[i] = s1[i1]
        s2_adapted[i] = s2[i2]
    return s1_adapted, s2_adapted


def normalized_dtw_score(
    s1: np.ndarray,
    s2: np.ndarray,
) -> float:
    """
    Calculate the normalized DTW score between two signals
    using the provided path.

    Parameters:
    - s1: array of shape (n_frames_1, n_features)
    - s2: array of shape (n_frames_2, n_features)
    """
    path_s1_s2, dist_s1_s2 = dtw_path(s1, s2)
    norm_dist = dist_s1_s2 / np.sqrt(len(path_s1_s2))
    s1_adapted, s2_adapted = signals_from_path(s1, s2, path_s1_s2)
    return norm_dist, s1_adapted, s2_adapted
