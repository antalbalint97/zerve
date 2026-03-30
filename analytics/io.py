import os
from pathlib import Path

import pandas as pd


DATA_PATH = "data/zerve_events.csv"
FEAT_PATH = "outputs/user_features_segmented.parquet"
OUTPUT_DIR = "outputs"


def ensure_output_dir(output_dir: str = OUTPUT_DIR) -> Path:
    path = Path(output_dir)
    path.mkdir(exist_ok=True)
    return path


def load_raw_events(
    data_path: str = DATA_PATH,
    parse_timestamps: bool = True,
    normalize_bool_strings: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(data_path, low_memory=False)
    if normalize_bool_strings:
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].replace({"IGAZ": True, "HAMIS": False})
    if parse_timestamps:
        for col in ["timestamp", "created_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    if "person_id" in df.columns:
        df["person_id"] = df["person_id"].astype("string")
    if "distinct_id" in df.columns:
        df["distinct_id"] = df["distinct_id"].astype("string")
    return df


def load_events(data_path: str = DATA_PATH) -> pd.DataFrame:
    df = load_raw_events(data_path=data_path)
    df = df.dropna(subset=["person_id", "timestamp"]).copy()
    df["person_id"] = df["person_id"].astype(str)
    return df


def load_features(feat_path: str = FEAT_PATH) -> pd.DataFrame:
    feat = pd.read_parquet(feat_path).copy()
    feat.index = feat.index.astype(str)
    return feat


def write_table(df: pd.DataFrame, path: str) -> None:
    ensure_output_dir(os.path.dirname(path) or OUTPUT_DIR)
    if path.endswith(".parquet"):
        df.to_parquet(path)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")


def merge_feature_columns(
    df: pd.DataFrame,
    feat: pd.DataFrame,
    columns: list[str],
    on: str = "person_id",
) -> pd.DataFrame:
    return df.merge(feat[columns].reset_index(), on=on, how="left")
