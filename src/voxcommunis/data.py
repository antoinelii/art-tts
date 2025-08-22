import pickle
from functools import cache
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .decoder import FeatureDecoder
from .io import read_alignment, read_manifest
from .utils import MyPathLike, unique_consecutive

SAMPLE_RATE = 16_000
ALIGNMENT_FREQ = 100  # in Hz
MODEL_FREQ = 50  # in Hz
SUBSAMPLE = ALIGNMENT_FREQ // MODEL_FREQ
LANGUAGES = {
    "ab": "Abkhaz",
    "ace": "Acehnese",
    "ady": "Adyghe",
    "af": "Afrikaans",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "arn": "Mapudungun",
    "as": "Assamese",
    "ast": "Asturian",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "bas": "Basaa",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bm": "Bambara",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "bxr": "Buryat",
    "byv": "Medumba",
    "ca": "Catalan",
    "cak": "Kaqchikel",
    "ckb": "Central Kurdish",
    "cnh": "Hakha Chin",
    "co": "Corsican",
    "crh": "Crimean Tatar",
    "cs": "Czech",
    "cv": "Chuvash",
    "cy": "Welsh",
    "da": "Danish",
    "dag": "Dagbani",
    "de": "German",
    "dsb": "Sorbian, Lower",
    "dv": "Dhivehi",
    "dyu": "Dioula",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "ewo": "Ewondo",
    "fa": "Persian",
    "ff": "Fulah",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "fuf": "Pular Guinea",
    "fy-NL": "Frisian",
    "ga-IE": "Irish",
    "gl": "Galician",
    "gn": "Guarani",
    "gom": "Goan Konkani",
    "gu-IN": "Gujarati",
    "guc": "Wayuunaiki",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hil": "Hiligaynon",
    "hr": "Croatian",
    "hsb": "Sorbian, Upper",
    "ht": "Haitian",
    "hu": "Hungarian",
    "hy-AM": "Armenian",
    "hyw": "Armenian Western",
    "ia": "Interlingua",
    "id": "Indonesian",
    "ie": "Interlingue",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "izh": "Izhorian",
    "ja": "Japanese",
    "jbo": "Lojban",
    "jv": "Javanese",
    "ka": "Georgian",
    "kaa": "Karakalpak",
    "kab": "Kabyle",
    "kbd": "Kabardian",
    "ki": "Kikuyu",
    "kk": "Kazakh",
    "km": "Khmer",
    "kmr": "Kurmanji Kurdish",
    "kn": "Kannada",
    "knn": "Konkani (Devanagari)",
    "ko": "Korean",
    "kpv": "Komi-Zyrian",
    "kw": "Cornish",
    "ky": "Kyrgyz",
    "lb": "Luxembourgish",
    "lg": "Luganda",
    "lij": "Ligurian",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "ltg": "Latgalian",
    "lv": "Latvian",
    "lzz": "Laz",
    "mai": "Maithili",
    "mdf": "Moksha",
    "mg": "Malagasy",
    "mhr": "Meadow Mari",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mni": "Meetei Lon",
    "mos": "Mossi",
    "mr": "Marathi",
    "mrj": "Hill Mari",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "myv": "Erzya",
    "nan-tw": "Taiwanese (Minnan)",
    "nb-NO": "Norwegian Bokmål",
    "nd": "IsiNdebele (North)",
    "ne-NP": "Nepali",
    "nhe": "Eastern Huasteca Nahuatl",
    "nhi": "Western Sierra Puebla Nahuatl",
    "nia": "Nias",
    "nl": "Dutch",
    "nn-NO": "Norwegian Nynorsk",
    "nr": "IsiNdebele (South)",
    "nso": "Northern Sotho",
    "ny": "Chinyanja",
    "nyn": "Runyankole",
    "oc": "Occitan",
    "om": "Afaan Oromo",
    "or": "Odia",
    "os": "Ossetian",
    "pa-IN": "Punjabi",
    "pap-AW": "Papiamento (Aruba)",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "quc": "K'iche'",
    "quy": "Quechua Chanka",
    "qvi": "Kichwa",
    "rm-sursilv": "Romansh Sursilvan",
    "rm-vallader": "Romansh Vallader",
    "ro": "Romanian",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "sah": "Sakha",
    "sat": "Santali (Ol Chiki)",
    "sc": "Sardinian",
    "scn": "Sicilian",
    "sco": "Scots",
    "sd": "Sindhi",
    "sdh": "Southern Kurdish",
    "shi": "Shilha",
    "si": "Sinhala",
    "sk": "Slovak",
    "skr": "Saraiki",
    "sl": "Slovenian",
    "snk": "Soninke",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "ss": "Siswati",
    "st": "Southern Sotho",
    "sv-SE": "Swedish",
    "sw": "Swahili",
    "syr": "Syriac",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "ti": "Tigrinya",
    "tig": "Tigre",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tn": "Setswana",
    "tok": "Toki Pona",
    "tr": "Turkish",
    "ts": "Xitsonga",
    "tt": "Tatar",
    "tw": "Twi",
    "ty": "Tahitian",
    "tyv": "Tuvan",
    "uby": "Ubykh",
    "udm": "Udmurt",
    "ug": "Uyghur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "ve": "Tshivenda",
    "vec": "Venetian",
    "vi": "Vietnamese",
    "vmw": "Emakhuwa",
    "vot": "Votic",
    "wep": "Westphalian",
    "wo": "Wolof",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "yue": "Cantonese",
    "zgh": "Tamazight",
    "zh-CN": "Chinese (China)",
    "zh-HK": "Chinese (Hong Kong)",
    "zh-TW": "Chinese (Taiwan)",
    "zu": "Zulu",
    "zza": "Zaza",
}


class FeatureTokenizer:
    """FeatureTokenizer is a class that handles tokenization and encoding of IPA
    (International Phonetic Alphabet) phones into feature representations using a
    FeatureDecoder.

    Methods
    -------
    __init__(unknown_mode, feature_decoder):
        Initializes the FeatureTokenizer with a specific unknown mode and a feature
        decoder.

    unknown_mode:
        Property to access the unknown mode setting.

    num_features:
        Property to get the total number of features as per feature decoder header.

    multilingual_mode:
        Property to check if multilingual mode is enabled in the feature decoder.

    ipa_to_features(ipa_phone):
        Converts an IPA phone to its representative features.

    encode(ipa_phones, counts):
        Encodes a sequence of IPA phones along with their counts into tensors.

    decode(tokens):
        Abstract method meant to decode feature tensors back into sequence of IPA
        phones.

    Notes
    -----
    Positive features are encoded as 1, negative features are encoded as 0, and zero
    features are encoded as 0 (if unknown mode is "no-unk") or 2 (otherwise).
    """

    def __init__(
        self,
        feature_decoder: FeatureDecoder,
    ) -> None:
        self._feat_decoder = feature_decoder
        self._ipa_to_feats = {
            seg: feats
            for seg, feats in zip(feature_decoder.segments, feature_decoder.features)
        }

    @property
    def num_features(self) -> int:
        return len(self._feat_decoder.header)

    @property
    def multilingual_mode(self) -> bool:
        return self._feat_decoder.multilingual_mode

    @cache
    def ipa_to_features(self, ipa_phone: str) -> tuple[tuple[str, ...], torch.Tensor]:
        """Get the representative form and the feature representation of an IPA phone.

        Parameters
        ----------
        ipa_phone: str
            The IPA phone to convert.

        Returns
        -------
        phone_strings : tuple of str
            A tuple with the representative phones.
        feature_tensor : torch.Tensor
            Tensor representing the features of the IPA phone.
        """
        rep_phones = self._feat_decoder.segment_to_representative(ipa_phone)
        rep_phones, vector = self._feat_decoder.canonical_representation(rep_phones)

        dtype = torch.float
        tensor = torch.from_numpy(vector).type(dtype)

        return rep_phones, tensor

    def encode(
        self, ipa_phones: tuple[str, ...], counts: tuple[int, ...]
    ) -> tuple[torch.Tensor, list[str]]:
        """Encode a sequence of IPA phones along with their counts into tensors.
        Parameters
        ----------
        ipa_phones : tuple of str
            A tuple of IPA symbols representing phones.
        counts : tuple of int
            A tuple of integers representing the repetition count of each phone.

        Returns
        -------
        feature_tensor : torch.Tensor
            The tensor containing the encoded features.
        phones : list of str
            A list of phone strings.
        """
        assert len(counts) == len(ipa_phones), (
            f"Length mismatch between the IPA phones ({len(ipa_phones)}) and counts "
            f"({len(counts)})"
        )
        vectors = []
        phones = []
        for phone, reps in zip(ipa_phones, counts):
            phs, vec = self.ipa_to_features(phone)
            if len(vec) == 1:
                vectors.append(vec.repeat(reps, 1))
                phones += [phs[0]] * reps
            else:
                boundaries = [round(i * reps / len(vec)) for i in range(len(vec) + 1)]
                lengths = [e - b for b, e in zip(boundaries[:-1], boundaries[1:])]
                vectors.append(vec.repeat_interleave(torch.tensor(lengths), dim=0))
                phones += [ph for ph, len_ in zip(phs, lengths) for _ in range(len_)]
        return torch.cat(vectors, dim=0), phones

    def decode(self, tokens: torch.Tensor) -> list[str]:
        """Decode feature tensors back into a sequence of IPA phones.

        Parameters
        ----------
        tokens : torch.Tensor
            The tensor containing the encoded features.

        Returns
        -------
        ipa_phones : list of str
            A list of IPA phone strings.
        """
        raise NotImplementedError


class PanPhonInventory:
    def __init__(self):
        with open("voxcommunis/correction_map.pickle", "rb") as fp:
            self._corrections = pickle.load(fp)

    def convert_to_ipa(self, panphon_phones: list[str] | str) -> str:
        if isinstance(panphon_phones, str):
            panphon_phones = panphon_phones.split(" ")
        panphon_phones = panphon_phones[::SUBSAMPLE]  # downsample to model freq
        panphon_phones = [
            self._corrections.get(phone, phone) for phone in panphon_phones
        ]
        return " ".join(panphon_phones)  # store as string to save memory


class PhoneticFeatureDataset(Dataset):
    def __init__(
        self,
        manifest_path: MyPathLike,
        alignment_path: MyPathLike,
        feature_tokenizer: FeatureTokenizer,
        separate_files: bool = False,
    ) -> None:
        super().__init__()
        self.feature_tokenizer = feature_tokenizer
        panphon_inventory = PanPhonInventory()
        if separate_files:
            manifests_list = sorted(list(Path(manifest_path).glob("*.tsv")))
            self.langs = [fp.stem for fp in manifests_list]
            self.lang_sizes = []
            self.manifest = []
            for man_path in manifests_list:
                man = read_manifest(man_path)
                self.manifest += list(man.items())
                self.lang_sizes.append(len(man))

            self.ipa_phones: dict[str, str] = {}
            for lang in self.langs:
                align_path = Path(alignment_path) / f"{lang}.align"
                alignments = read_alignment(align_path)
                self.ipa_phones.update(
                    {
                        file: panphon_inventory.convert_to_ipa(_align)
                        for file, _align in alignments.items()
                    }
                )
        else:
            manifest = read_manifest(manifest_path)
            self.manifest = list(manifest.items())
            alignments = read_alignment(alignment_path)
            assert feature_tokenizer.multilingual_mode
            self.ipa_phones = {
                file: panphon_inventory.convert_to_ipa(_align)
                for file, _align in alignments.items()
            }

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Sequence[str]]:
        if idx >= len(self.manifest):
            raise IndexError(f"Index {idx} out of range")

        file_id, (path, num_samples) = self.manifest[idx]
        phones = self.ipa_phones[file_id].split(" ")  # parsed as list of str
        phones, counts = unique_consecutive(phones, return_counts=True)
        counts1 = [1 for _ in phones]  # create counts of 1 for each phone
        phon_features, phones = self.feature_tokenizer.encode(phones, counts1)
        # Add silence trait as an additional 25th dimension
        sil_trait = (phon_features == 0).all(
            axis=1
        ) * 2 - 1  # 1 for sil, -1 for non-sil
        phon_features = torch.concat([phon_features, sil_trait.unsqueeze(1)], dim=1)
        # add counts to the as an additional 26th dimension
        counts = torch.as_tensor(counts, dtype=torch.float).unsqueeze(1)
        phon_features = torch.concat([phon_features, counts], dim=1)

        return phon_features, phones
