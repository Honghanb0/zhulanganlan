"""Leakage-aware security event feature engineering for model v3."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


RAW_CATEGORICAL = [
    "pipeline", "product_name", "vendor_name", "src_host", "dst_host", "username"
]
FREQUENCY_KEYS = [
    "src_ip", "dst_ip", "username", "src_host", "dst_host", "product_name",
    "src_dst_pair", "host_user_pair", "product_pipeline_pair",
]
HASH_KEYS = [
    "src_ip", "dst_ip", "username", "src_dst_pair", "host_user_pair",
    "message_exact", "message_prefix", "message_suffix",
]
SECURITY_PATTERNS = {
    "kw_deny_reject": r"\b(?:deny|denied|reject|blocked|drop)\b",
    "kw_auth_failure": r"\b(?:fail(?:ed|ure)?|invalid|unauthori[sz]ed|bad password)\b",
    "kw_malware": r"\b(?:malware|ransom|trojan|virus|exploit|payload|shellcode)\b",
    "kw_email": r"\b(?:phish|spam|attachment|sender|recipient|smtp)\b",
    "kw_privilege": r"\b(?:sudo|admin|privilege|elevat|root)\w*\b",
    "kw_script": r"\b(?:powershell|cmd\.exe|wscript|bash|curl|wget)\b",
    "kw_network": r"\b(?:firewall|connection|protocol|src|dst|dport|sport)\b",
    "kw_cloud": r"\b(?:cloudtrail|aws|assumerole|securitygroup|vpc)\b",
}


def _series(df: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    if name in df:
        return df[name].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def stable_hash(values: pd.Series, buckets: int) -> np.ndarray:
    return (pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64) % buckets).astype(np.int32)


def make_group_ids(df: pd.DataFrame) -> np.ndarray:
    """Keep identical message features, including empty messages, in one fold."""
    message = _series(df, "message_sanitized")
    return pd.util.hash_pandas_object(message, index=False).to_numpy(dtype=np.uint64)


@dataclass
class SecurityFeaturePipeline:
    hash_buckets: int = 16384
    rare_min_count: int = 5
    window_seconds: tuple[int, ...] = (300, 3600, 86400)
    include_absolute_calendar_features: bool = True
    category_maps_: dict = field(default_factory=dict)
    frequency_maps_: dict = field(default_factory=dict)
    feature_names_: list[str] = field(default_factory=list)
    categorical_features_: list[str] = field(default_factory=list)
    feature_importance_: dict = field(default_factory=dict)
    fit_rows_: int = 0

    def _base(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
        n = len(df)
        raw = {c: _series(df, c) for c in set(RAW_CATEGORICAL + [
            "src_ip", "dst_ip", "message_sanitized", "src_host", "dst_host", "username",
            "pipeline", "product_name", "vendor_name"
        ])}
        raw["src_dst_pair"] = raw["src_ip"] + "\x1f" + raw["dst_ip"]
        raw["host_user_pair"] = raw["src_host"] + "\x1f" + raw["username"]
        raw["product_pipeline_pair"] = raw["product_name"] + "\x1f" + raw["pipeline"]
        msg = raw["message_sanitized"].str.lower()
        raw["message_exact"] = msg
        raw["message_prefix"] = msg.str.slice(0, 192)
        raw["message_suffix"] = msg.str.slice(-128)

        timestamp = pd.to_numeric(df.get("timestamp", pd.Series(np.nan, index=df.index)), errors="coerce")
        timestamp = timestamp.replace([np.inf, -np.inf], np.nan)
        dt = pd.to_datetime(timestamp, unit="s", errors="coerce")
        hour = dt.dt.hour.fillna(-1).astype(np.float32)
        dow = dt.dt.dayofweek.fillna(-1).astype(np.float32)
        out = pd.DataFrame(index=df.index)
        if self.include_absolute_calendar_features:
            out["timestamp_year"] = dt.dt.year.fillna(-1).astype(np.float32)
            out["timestamp_month"] = dt.dt.month.fillna(-1).astype(np.float32)
            out["timestamp_day"] = dt.dt.day.fillna(-1).astype(np.float32)
        out["timestamp_hour"] = hour
        out["timestamp_dow"] = dow
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)
        out["src_port"] = pd.to_numeric(df.get("src_port", -1), errors="coerce").fillna(-1).astype(np.float32)
        out["message_length_log"] = np.log1p(raw["message_sanitized"].str.len()).astype(np.float32)
        out["message_lines_log"] = np.log1p(raw["message_sanitized"].str.count(r"[\r\n]")).astype(np.float32)
        out["placeholder_count_log"] = np.log1p(
            raw["message_sanitized"].str.count(r"(?:USER|ORG|CRED|HOST|IP)-")
        ).astype(np.float32)
        out["src_private"] = raw["src_ip"].str.match(r"^(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)").astype(np.float32)
        out["dst_private"] = raw["dst_ip"].str.match(r"^(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)").astype(np.float32)
        out["src_missing"] = raw["src_ip"].eq("").astype(np.float32)
        out["dst_missing"] = raw["dst_ip"].eq("").astype(np.float32)
        out["user_missing"] = raw["username"].eq("").astype(np.float32)
        out["product_missing"] = raw["product_name"].eq("").astype(np.float32)
        for name, pattern in SECURITY_PATTERNS.items():
            out[name] = msg.str.contains(pattern, regex=True, na=False).astype(np.float32)

        # Stable categorical hashes preserve entity and sequence relationships without huge one-hot matrices.
        for key in HASH_KEYS:
            name = f"cat_hash_{key}"
            out[name] = stable_hash(raw[key], self.hash_buckets)

        # Batch window counts expose bursts/scans. They use no labels and are valid for offline batch inference.
        seconds = timestamp.fillna(-1).astype(np.int64)
        for window in self.window_seconds:
            bucket = seconds // int(window)
            for key in ("src_ip", "dst_ip", "username"):
                name = f"window_{window}_{key}_count_log"
                counts = pd.DataFrame({"entity": raw[key], "bucket": bucket}).groupby(
                    ["entity", "bucket"], sort=False, dropna=False
                )["entity"].transform("size")
                out[name] = np.log1p(counts).astype(np.float32)

        # Previous-event gaps are causal sequence features. One timestamp sort serves all entities.
        order = np.argsort(seconds.to_numpy(), kind="stable")
        sorted_seconds = seconds.iloc[order]
        for key in ("src_ip", "username", "src_host"):
            sorted_entity = raw[key].iloc[order]
            prev = sorted_seconds.groupby(sorted_entity, sort=False).shift(1)
            delta = (sorted_seconds - prev).clip(lower=0).fillna(1e9)
            restored = np.empty(n, dtype=np.float32)
            restored[order] = np.log1p(delta.to_numpy()).astype(np.float32)
            out[f"previous_gap_{key}_log"] = restored
        return out, raw

    def fit(self, df: pd.DataFrame) -> "SecurityFeaturePipeline":
        _, raw = self._base(df)
        self.fit_rows_ = len(df)
        self.category_maps_ = {}
        for col in RAW_CATEGORICAL:
            counts = raw[col].value_counts(dropna=False)
            keep = counts[counts >= self.rare_min_count].index
            self.category_maps_[col] = {value: i for i, value in enumerate(keep, start=1)}
        self.frequency_maps_ = {}
        total = max(len(df), 1)
        for key in FREQUENCY_KEYS:
            self.frequency_maps_[key] = (raw[key].value_counts(dropna=False) / total).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out, raw = self._base(df)
        categorical = []
        for col in RAW_CATEGORICAL:
            name = f"cat_{col}"
            out[name] = raw[col].map(self.category_maps_.get(col, {})).fillna(0).astype(np.int32)
            categorical.append(name)
        categorical.extend([f"cat_hash_{key}" for key in HASH_KEYS])
        for key in FREQUENCY_KEYS:
            mapped = raw[key].map(self.frequency_maps_.get(key, {})).fillna(0.0)
            # Convert the fitted relative frequency back to its training-fold count.
            # It must not depend on inference batch size.
            out[f"frequency_{key}_log"] = np.log1p(
                mapped.astype(np.float64) * max(self.fit_rows_, 1)
            ).astype(np.float32)
            out[f"unseen_{key}"] = mapped.eq(0).astype(np.float32)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(-1)
        self.feature_names_ = list(out.columns)
        self.categorical_features_ = categorical
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
