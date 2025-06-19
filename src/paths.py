from pathlib import Path

SRC_DIR = Path(__file__).parent

# local computer specific
LOGS_DIR = SRC_DIR / "../logs"
DATA_DIR = SRC_DIR / "../data"

WAVS_DIR = DATA_DIR / "LJSpeech-1.1" / "wavs"
ENCODED_AUDIO_EN_DIR = DATA_DIR / "LJSpeech-1.1" / "encoded_audio_en"
EMASRC_DIR = ENCODED_AUDIO_EN_DIR / "emasrc"
SPK_EMB_DIR = ENCODED_AUDIO_EN_DIR / "spk_emb"

# global specific
CKPT_DIR = SRC_DIR / "ckpt"
CONFIGS_DIR = SRC_DIR / "configs"
MODELS_DIR = SRC_DIR / "model"
RESOURCES_DIR = SRC_DIR / "resources"
SCRIPTS_DIR = SRC_DIR / "scripts"
TEXT_DIR = SRC_DIR / "text"

FILELISTS_DIR = RESOURCES_DIR / "filelists"
LJLISTS_DIR = FILELISTS_DIR / "ljspeech"
