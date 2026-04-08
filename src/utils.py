from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for folder in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
