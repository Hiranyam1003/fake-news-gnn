from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
DATASET_DIR = RAW_DIR / "rumor_detection_acl2017"
PROCESSED_DIR = ROOT / "data" / "processed"
RUNS_DIR = ROOT / "runs"
DATA_URL = "https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=1"

DATASETS = ("twitter15", "twitter16")
MODES = ("content", "structure", "joint")
STAGES = ("prototype", "beta")

LABELS = ["non-rumor", "false", "true", "unverified"]
VERDICT = {"non-rumor": "Real", "true": "Real", "false": "Fake", "unverified": "Needs review"}

SEED = 42
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

STAGE_CONFIG = {
    "prototype": dict(
        text_encoder="tfidf", root_enhance=True, hidden=64, layers=2, dropout=0.3, lr=1e-3, weight_decay=1e-4,
        epochs=100, patience=10, batch_size=32, folds=1, truncate_train=False,
        early_k=[5, 10, None], early_hours=[],
    ),
    "beta": dict(
        text_encoder="tfidf", root_enhance=False, hidden=128, layers=3, dropout=0.3, lr=1e-3, weight_decay=1e-4,
        epochs=150, patience=15, batch_size=32, folds=5, truncate_train=True,
        early_k=[5, 10, 20, 50, None], early_hours=[0, 1, 4, 12, 24],
    ),
}


def run_dir(stage, dataset, mode):
    return RUNS_DIR / stage / dataset / mode
