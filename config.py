"""Shared paths and submission invariants for the current v3 pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models_v3"
OUTPUT_DIR = ROOT_DIR / "output"

TRAIN_FILE = DATA_DIR / "train.parquet"
VALID_FILE = DATA_DIR / "valid_input.parquet"
TEST_FILE = DATA_DIR / "test_input.csv"
# Best submission score confirmed by the official platform. Unscored rules are
# disabled by default because the latest OOD candidate regressed online.
VERIFIED_BASE_SUBMISSION = ROOT_DIR / "solution" / "0.96802513.csv"
TARGET_SUBMISSION = OUTPUT_DIR / "res_0.97297170_target.csv"

# 官方测试集是 CSV 中的 55,837 个样本（加表头后 CSV 物理行数为 55,838）。
# data/test_input.parquet 有 4,234,041 行，是另一批数据，不能用于比赛提交。
EXPECTED_TEST_ROWS = 55_837

LABEL_NAMES = ("benign", "suspicious", "malicious")
