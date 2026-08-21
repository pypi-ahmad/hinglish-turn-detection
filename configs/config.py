"""Shared config for data prep, training, eval, export, and the demo app."""

from pathlib import Path

# configs/config.py lives one level below the project root, so ROOT needs
# `.parent.parent`, not `.parent` (a plain `.parent` would silently create a
# `configs/data/` directory instead of a project-root `data/`).
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_CACHE_DIR = DATA_DIR / "raw"
SUBSET_DIR = DATA_DIR / "subset"
CHECKPOINT_DIR = ROOT / "checkpoints"
ONNX_DIR = ROOT / "onnx"

for d in (DATA_DIR, RAW_CACHE_DIR, SUBSET_DIR, CHECKPOINT_DIR, ONNX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- audio / model ---
SAMPLE_RATE = 16_000
MAX_DURATION_S = 8  # cap on *real* speech content we keep, per the smart-turn convention
MAX_REAL_SAMPLES = MAX_DURATION_S * SAMPLE_RATE  # 128,000

# Pipecat's official Smart Turn v3.2 resizes Whisper's learned positional
# embedding table to an 8-second window: 800 mel frames -> 400 encoder
# positions. Short clips are left-padded so their ending stays aligned.
WHISPER_WINDOW_S = MAX_DURATION_S
WHISPER_WINDOW_SAMPLES = MAX_REAL_SAMPLES  # 128,000
WHISPER_MEL_FRAMES = 800
WHISPER_ENCODER_HIDDEN_LEN = 400
BASE_MODEL_NAME = "openai/whisper-tiny"

# --- source datasets (Pipecat's own smart-turn training data) ---
TRAIN_DATASET_REPO = "pipecat-ai/smart-turn-data-v3.2-train"
TEST_DATASET_REPO = "pipecat-ai/smart-turn-data-v3.2-test"

# language of primary interest + the language it code-switches with
PRIMARY_LANGUAGE = "hin"  # Hindi
BRIDGE_LANGUAGE = "eng"  # English (Hinglish = hin<->eng code-switching)

# subset sizing (streamed + filtered from the full 41GB/270k-row dataset,
# so we never download the whole thing).
# Hindi is only ~4.5% of rows with no sort order we can exploit, so an exhaustive
# scan for "all hin rows" means reading close to the full 41GB. We instead cap
# the number of rows *scanned* and keep whatever hin/eng rows appear inside that
# budget -- a deliberate bandwidth/time trade-off, documented in the README.
ROW_SCAN_BUDGET_TRAIN = 120_000  # ~44% of the 270,946-row train split
ROW_SCAN_BUDGET_TEST = 40_000
MAX_PRIMARY_ROWS = 20_000       # "hin" rows, capped (expect to land well under this)
MAX_BRIDGE_ROWS = 20_000        # matched-size "eng" rows (code-switch augmentation source + baseline retention)
MAX_ROWS_PER_OTHER_LANG = 150   # opportunistic only -- whatever shows up inside the same bounded scan

# --- training ---
BATCH_SIZE = 24
GRAD_ACCUM_STEPS = 4  # effective batch 96, fits an 8GB laptop GPU with fp16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 5e-5
NUM_EPOCHS = 6
WARMUP_RATIO = 0.2
WEIGHT_DECAY = 0.01
FREEZE_ENCODER_LAYERS = 0  # whisper-tiny has 4 encoder layers total

# --- Hinglish augmentation ---
# Common Hindi/Hinglish discourse fillers + pause markers to splice near
# utterance boundaries (per the challenge brief: "filler words and pauses").
HINGLISH_FILLERS = [
    "um", "uh", "matlab", "actually", "tho", "toh", "yaar", "bas",
    "wait", "ek second", "haan", "vo", "voh", "acha", "umm", "hmm",
    "you know", "like", "so",
]
AUGMENT_FRACTION = 0.35  # fraction of primary+bridge rows to turn into synthetic filler/pause examples
TTS_VOICES = [
    "hi-IN-SwaraNeural", "hi-IN-MadhurNeural",       # Hindi
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",     # Indian-accented English
]
