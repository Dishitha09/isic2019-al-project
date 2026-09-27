"""
Data preparation for the ISIC 2019 preliminary baseline.

Handles:
- Reading the ground-truth CSV and dropping true-UNK rows correctly
  (instead of letting them silently collapse into "MEL").
- Merging with metadata to get lesion_id.
- A lesion-level (not image-level) train/val/test split, so images of the
  same lesion never appear in more than one split -- this is what makes the
  split "leakage-safe."
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ISIC 2019's ground-truth CSV has 9 columns: our 8 target diagnoses plus
# UNK ("none of the 8 confirmed"). UNK is NOT a classification target --
# we exclude those images entirely rather than default them into a class.
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
UNK_COLUMN = "UNK"
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


def prepare_dataframe(gt_csv, metadata_csv, image_dir):
    gt = pd.read_csv(gt_csv)
    meta = pd.read_csv(metadata_csv)

    missing = [c for c in CLASSES if c not in gt.columns]
    if missing:
        raise ValueError(f"Missing expected class columns in ground-truth CSV: {missing}")

    # Drop true-UNK rows: a row is "known" only if exactly one of our 8
    # target columns is 1. If we just selected gt[["image"] + CLASSES]
    # first, a genuine UNK row (all zeros across our 8 columns) would go
    # all-zero, and idxmax() on an all-zero row returns the FIRST column
    # ("MEL") -- silently mislabeling unknowns as melanoma.
    if UNK_COLUMN in gt.columns:
        n_before = len(gt)
        known_mask = gt[CLASSES].sum(axis=1) > 0
        n_unk = int((~known_mask).sum())
        gt = gt[known_mask].reset_index(drop=True)
        print(f"Dropped {n_unk} UNK-labeled images out of {n_before} "
              f"(kept {len(gt)} with a confirmed diagnosis).")
    else:
        print("No UNK column found -- assuming all rows already have a "
              "confirmed label among CLASSES.")

    df = gt[["image"] + CLASSES].copy()
    df["label"] = df[CLASSES].idxmax(axis=1)
    df["label_idx"] = df["label"].map(CLASS_TO_IDX)

    # Merge in lesion_id for the leakage-safe split. If it's missing, warn
    # loudly instead of quietly degrading to a plain image-level split.
    meta_cols = ["image"]
    has_lesion_id = "lesion_id" in meta.columns
    if has_lesion_id:
        meta_cols.append("lesion_id")

    df = df.merge(meta[meta_cols], on="image", how="left")

    if has_lesion_id:
        n_missing_lesion = df["lesion_id"].isna().sum()
        df["lesion_id"] = df["lesion_id"].fillna(df["image"])
        print(f"lesion_id found in metadata. {n_missing_lesion} images had "
              f"no lesion_id and are treated as their own single-image lesion.")
    else:
        df["lesion_id"] = df["image"]
        print("WARNING: metadata CSV has no 'lesion_id' column -- the "
              "leakage-safe split cannot group same-lesion images. Check "
              "you're pointing at ISIC_2019_Training_Metadata.csv.")

    image_dir = Path(image_dir)

    def resolve_path(image_id):
        p1 = image_dir / f"{image_id}.jpg"
        p2 = image_dir / image_id
        if p1.exists():
            return str(p1)
        if p2.exists():
            return str(p2)
        return str(p1)

    df["filepath"] = df["image"].apply(resolve_path)

    missing_files = (~df["filepath"].apply(os.path.exists)).sum()
    if missing_files:
        raise FileNotFoundError(
            f"{missing_files} image files were not found. "
            f"Check --image_dir and filename extensions."
        )

    return df


def lesion_level_split(df, seed=42):
    """Split at the LESION level so no lesion's images span more than one
    of train/val/test. Stratified on diagnosis to keep class proportions
    roughly consistent across splits."""
    group_df = (
        df.groupby("lesion_id", as_index=False)
          .agg(label=("label", lambda s: s.mode().iloc[0]))
    )

    train_groups, temp_groups = train_test_split(
        group_df, test_size=0.30, random_state=seed, stratify=group_df["label"]
    )
    val_groups, test_groups = train_test_split(
        temp_groups, test_size=0.50, random_state=seed, stratify=temp_groups["label"]
    )

    train_ids = set(train_groups["lesion_id"])
    val_ids = set(val_groups["lesion_id"])
    test_ids = set(test_groups["lesion_id"])

    train_df = df[df["lesion_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["lesion_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["lesion_id"].isin(test_ids)].reset_index(drop=True)

    assert set(train_df["lesion_id"]).isdisjoint(set(val_df["lesion_id"]))
    assert set(train_df["lesion_id"]).isdisjoint(set(test_df["lesion_id"]))
    assert set(val_df["lesion_id"]).isdisjoint(set(test_df["lesion_id"]))

    # Surface per-class LESION counts per split -- catches a rare class
    # (e.g. DF, VASC) ending up with too few lesions in val/test to trust
    # its per-class F1/AUC.
    print("\nPer-class LESION counts by split:")
    counts_table = pd.DataFrame({
        "train": train_groups["label"].value_counts(),
        "val": val_groups["label"].value_counts(),
        "test": test_groups["label"].value_counts(),
    }).reindex(CLASSES).fillna(0).astype(int)
    print(counts_table.to_string())

    thin = counts_table[(counts_table["val"] < 5) | (counts_table["test"] < 5)]
    if not thin.empty:
        print("\nWARNING: these classes have <5 lesions in val or test -- "
              "their per-class metrics will be noisy:")
        print(thin.to_string())

    return train_df, val_df, test_df
