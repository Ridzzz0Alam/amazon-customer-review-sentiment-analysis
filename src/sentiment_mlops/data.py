"""Data loading, labelling, de-duplication and the group aware split."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REQUIRED_COLUMNS = ["Product Name", "Brand Name", "Price", "Rating", "Reviews", "Review Votes"]


def normalise(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace. Used as the duplicate key."""
    text = re.sub(r"[\W_]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def load_raw(path: str, sample_size: int | None, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
    return df.reset_index(drop=True)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty reviews and 3 star reviews. 4-5 stars -> 1 and 1-2 stars -> 0."""
    out = df.copy()
    out["Reviews"] = out["Reviews"].fillna("").astype(str).str.strip()
    out = out[out["Reviews"] != ""]
    out = out[out["Rating"] != 3].copy()
    out["label"] = (out["Rating"] >= 4).astype(int)
    out["text_key"] = out["Reviews"].map(normalise)
    return out[out["text_key"] != ""].reset_index(drop=True)


def collapse_duplicates(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Keep one row per normalised text with its majority label.

    Texts whose majority share is below `threshold` are dropped as conflicting.
    The kept row uses the product name where the text appears most often.
    """
    stats = df.groupby("text_key").agg(pos_share=("label", "mean"), copies=("label", "size"))
    stats["agreement"] = stats["pos_share"].where(stats["pos_share"] >= 0.5, 1 - stats["pos_share"])
    stats["majority"] = (stats["pos_share"] >= 0.5).astype(int)
    keep_keys = stats[stats["agreement"] >= threshold]

    merged = df.merge(keep_keys[["majority", "copies"]], left_on="text_key", right_index=True)
    merged = merged[merged["label"] == merged["majority"]]
    product_counts = merged.groupby(["text_key", "Product Name"]).size().rename("product_count")
    merged = merged.join(product_counts, on=["text_key", "Product Name"])
    merged = merged.sort_values(["text_key", "product_count"], ascending=[True, False], kind="stable")
    out = merged.drop_duplicates("text_key").drop(columns=["majority", "product_count"])
    return out.reset_index(drop=True)


@dataclass
class Split:
    dev: pd.DataFrame
    test: pd.DataFrame


def group_split(df: pd.DataFrame, test_size: float, random_state: int) -> Split:
    """Hold out whole product names so the test set measures unseen products."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    dev_idx, test_idx = next(splitter.split(df, df["label"], groups=df["Product Name"]))
    dev, test = df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    assert not set(dev["Product Name"]) & set(test["Product Name"]), "product leakage"
    assert not set(dev["text_key"]) & set(test["text_key"]), "text leakage"
    return Split(dev=dev, test=test)


def prepare(path: str, sample_size: int | None, random_state: int, threshold: float, test_size: float) -> Split:
    raw = load_raw(path, sample_size, random_state)
    labelled = add_labels(raw)
    clean = collapse_duplicates(labelled, threshold)
    return group_split(clean, test_size, random_state)
