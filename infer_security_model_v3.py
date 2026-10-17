"""Batch inference for the v3 hierarchical security-event model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import EXPECTED_TEST_ROWS, MODEL_DIR, OUTPUT_DIR, TEST_FILE
from security_metrics import combine_probabilities, threshold_predict


def predict_frame(df: pd.DataFrame, artifact: dict) -> tuple[np.ndarray, np.ndarray]:
    x = artifact["feature_pipeline"].transform(df)
    models = artifact["models"]
    multi = models["multiclass"].predict_proba(x)
    threat = models["threat"].predict_proba(x)[:, 1]
    subtype = models["subtype"].predict_proba(x)[:, 1]
    policy = artifact["policy"]
    proba = combine_probabilities(
        multi, threat, subtype, policy["threat_weight"], policy["subtype_weight"]
    )
    pred = threshold_predict(proba, policy["threat_threshold"], policy["malicious_threshold"])
    return pred, proba


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=TEST_FILE,
                        help="Input event file; defaults to the official 55,837-row data/test_input.csv")
    parser.add_argument("output", nargs="?", default=str(OUTPUT_DIR / "v3_test_predictions.csv"),
                        help="Output CSV path")
    parser.add_argument("--model", default=str(MODEL_DIR / "security_model_v3.joblib"))
    parser.add_argument("--include-probabilities", action="store_true")
    parser.add_argument("--competition", action="store_true",
                        help="Require the exact official test row count and two-column submission schema")
    args = parser.parse_args()
    artifact = joblib.load(args.model)
    if args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_parquet(args.input)
    if "event_id" not in df:
        raise ValueError("input must contain event_id")
    if args.competition:
        if len(df) != EXPECTED_TEST_ROWS:
            raise ValueError(
                f"competition input must contain {EXPECTED_TEST_ROWS:,} samples, got {len(df):,}. "
                "Use data/test_input.csv, not data/test_input.parquet."
            )
        if df.event_id.isna().any() or df.event_id.duplicated().any():
            raise ValueError("competition input event_id must be complete and unique")
        if args.include_probabilities:
            raise ValueError("competition CSV may only contain event_id and pred_label")
    pred, proba = predict_frame(df, artifact)
    names = np.asarray(artifact["label_names"])
    result = pd.DataFrame({"event_id": df.event_id, "pred_label": names[pred]})
    if args.include_probabilities:
        result[["proba_benign", "proba_suspicious", "proba_malicious"]] = proba
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    if args.competition:
        if list(result.columns) != ["event_id", "pred_label"] or len(result) != EXPECTED_TEST_ROWS:
            raise AssertionError("invalid competition submission schema or row count")
    print(result.pred_label.value_counts().to_string())
    print(f"saved {len(result):,} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
