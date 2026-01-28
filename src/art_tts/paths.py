from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent

# local computer specific
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

WAVS_DIR = DATA_DIR / "LJSpeech-1.1" / "wavs"
ENCODED_AUDIO_EN_DIR = DATA_DIR / "LJSpeech-1.1" / "encoded_audio_en"
EMASRC_DIR = ENCODED_AUDIO_EN_DIR / "emasrc"
SPK_EMB_DIR = ENCODED_AUDIO_EN_DIR / "spk_emb"

# global specific
CKPT_DIR = PROJECT_ROOT / "ckpt"
RESOURCES_DIR = PROJECT_ROOT / "resources"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

ART_TTS_DIR = SRC_DIR / "art_tts"
CONFIGS_DIR = ART_TTS_DIR / "configs"
MODELS_DIR = ART_TTS_DIR / "model"
TEXT_DIR = ART_TTS_DIR / "text"

FILELISTS_DIR = RESOURCES_DIR / "filelists"
LJLISTS_DIR = FILELISTS_DIR / "ljspeech"
