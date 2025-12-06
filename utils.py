from pathlib import Path

BASE_DIR = Path.cwd()
BOOKS_DIR = BASE_DIR / "examples" / "books"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "model_out"   # куда сохранять чекпоинты и модель
LOG_DIR = BASE_DIR / "logs"

# Model & training
MODEL_NAME = "cointegrated/rubert-tiny"
MAX_SEQ_LEN = 256
BATCH_SIZE = 16
NUM_EPOCHS = 3
LR = 2e-5
TOP_K_LABEL = 3   # при разметке: top-K per chapter -> label=1
TOP_N_SUMMARY = 3 # при инференсе: top-N предложений
RANDOM_SEED = 42
