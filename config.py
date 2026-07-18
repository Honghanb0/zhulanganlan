import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, 'train.parquet')
VALID_FILE = os.path.join(DATA_DIR, 'valid_input.parquet')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'res.csv')

LABEL_MAPPING = {'benign': 0, 'suspicious': 1, 'malicious': 2}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

SEED = 42

MAX_FEATURES = 2000
MAX_SEQ_LEN = 256
EMBEDDING_DIM = 128

NUM_EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

CLASS_WEIGHTS = {
    0: 1.0,
    1: 40.0,
    2: 15.0
}

SAMPLE_SIZE = 100000