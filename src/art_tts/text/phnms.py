import numpy as np
from text.converters import diphtongues_ipa


def build_phnm3(
    phonemes,
    t_boundaries,
):
    """
    Builds a phnm3 object.
    Args:
        phonemes: list of IPA phonemes
        t_boundaries: list of time boundaries for each phoneme
        length should be len(phonemes) + 1
    Returns:
        phnm3: np.ndarray, shape (M, 3) with (start, end, phoneme)
        where M is the number of phonemes after alignment
    """
    assert len(t_boundaries) == len(phonemes) + 1, (
        "t_boundaries should be of length len(phonemes) + 1"
        f"but got {len(t_boundaries)} and {len(phonemes)}"
    )

    phnm3 = []
    for i, phnm in enumerate(phonemes):
        phnm3.append((t_boundaries[i], t_boundaries[i + 1], phnm))
    phnm3 = np.array(phnm3, dtype=[("start", "f4"), ("end", "f4"), ("phoneme", "U10")])
    return phnm3


def get_phnms_from_phnm3(phnm3, merge_diphtongues):
    if merge_diphtongues:
        phnms = [e[2] for e in phnm3]
    else:  # divide diphtongues into two phonemes to match nb of phonemes of input data
        phnms = []
        for e in phnm3:
            phone = e[2]
            if phone not in diphtongues_ipa:
                phnms.append(phone)
            else:
                phnms.append(phone[0])
                phnms.append(phone[1])
    return phnms


def get_pred_phnm3(phnm3, phnm_map, merge_diphtongues=False):
    """
    Builds a phnm3 object.
    Aligns ground truth phonemes with model phoneme lengths.
    (using phnm3 times disaligns with predicted signal
    due to delta_t to nb frames rounding)

    Args:
        phnm3: np.ndarray, shape (N, 3) with (start, end, phoneme)
        phnm_map: np.ndarray, shape (T,) with matching phnm number for each frame
        merge_diphtongues: bool, whether to merge diphtongues into single phoneme
    Returns:
        phnm3_ada: np.ndarray, shape (M, 3) with (start, end, phoneme)
        where M is the number of phonemes after alignment
    """

    phnms = get_phnms_from_phnm3(phnm3, merge_diphtongues)

    art_sr = 50
    t_end = phnm_map.shape[0] / art_sr
    t_boundaries = list((np.where(np.diff(phnm_map) == 1)[0] + 1) / art_sr)
    t_boundaries = [0] + t_boundaries + [t_end]

    phnm3_ada = build_phnm3(phnms, t_boundaries)
    return phnm3_ada


def get_lengths_from_phnm3(phnm3, merge_diphtongues: bool = False) -> np.ndarray:
    if merge_diphtongues:
        durations = [e[1] - e[0] for e in phnm3]
    else:
        durations = []
        for start, end, phone in phnm3:
            if phone in diphtongues_ipa:
                mid = (end + start) / 2
                durations.append(mid - start)
                durations.append(end - mid)
            else:
                durations.append(end - start)
    durations = np.array(durations, dtype=np.float32)
    return durations
