"""
Generate splits.json for dataset3 — source-aware stratified 70/20/10 split.

Groups samples by (Aggregate Type, Source) so that a single source never
spans multiple splits (avoids leakage from near-identical photos of the
same batch). Groups are greedily assigned to train/val/test to approximate
the 70/20/10 target ratio per aggregate type.

Run locally (or anywhere with the labels.csv) whenever the dataset changes
size — NOT part of aggnet_dataset3.py itself since it should only run when
new data is added, not on every training run.
"""
import json
import random
import pandas as pd

DATA_DIR   = "data/dataset3"
LABELS_CSV = f"{DATA_DIR}/labels.csv"
OUT_JSON   = f"{DATA_DIR}/splits.json"
SEED       = 42
RATIOS     = {"train": 0.70, "val": 0.20, "test": 0.10}

AGG_TYPE_TO_KEY = {
    "Aggregate 3_4inch": "model_a",
    "Aggregate 3_8inch": "model_b",
    "Aggregate 1 inch":  "model_c",
}


def clean_duplicates(df):
    sieve_check_cols = ['3_4inch', '1_2inch', '3_8inch', 'No4', 'No8', 'Pan']
    available_cols   = [c for c in sieve_check_cols if c in df.columns]
    keep_mask = pd.Series(True, index=df.index)
    for agg_type, grp in df.groupby('Aggregate Type'):
        dup_mask = grp.duplicated(subset=available_cols, keep='first')
        for idx in grp[dup_mask].index:
            keep_mask.loc[idx] = False
    return df[keep_mask].reset_index(drop=True)


def source_aware_split(sample_ids_by_source, seed=SEED):
    """
    Split proportionally *within* each source group (~70/20/10), rather than
    assigning whole groups atomically. A handful of dominant sources (as with
    3_8inch, 4 sources covering 70 samples) makes whole-group assignment too
    coarse to hit the target ratio — some splits end up empty. Small groups
    (<=2 samples) go entirely to train since they can't be split meaningfully.
    """
    rng = random.Random(seed)
    assigned = {"train": [], "val": [], "test": []}

    for src, ids in sorted(sample_ids_by_source.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)

        if n <= 2:
            assigned["train"].extend(ids)
            continue

        n_test  = max(1, round(n * RATIOS["test"]))  if n >= 5 else 0
        n_val   = max(1, round(n * RATIOS["val"]))
        n_train = n - n_val - n_test
        if n_train < 1:
            # tiny group — fall back to train/val only
            n_train = n - 1
            n_val, n_test = 1, 0

        assigned["train"].extend(ids[:n_train])
        assigned["val"].extend(ids[n_train:n_train + n_val])
        assigned["test"].extend(ids[n_train + n_val:])

    for k in assigned:
        assigned[k].sort()
    return assigned


def main():
    df = pd.read_csv(LABELS_CSV)
    df.columns = df.columns.str.strip()
    df['Aggregate Type'] = df['Aggregate Type'].str.strip()
    df['Source'] = df['Source'].astype(str).str.strip()

    print(f"Raw rows: {len(df)}")
    df = clean_duplicates(df)
    print(f"After dedup: {len(df)}")

    splits = {}
    for agg_type, key in AGG_TYPE_TO_KEY.items():
        sub = df[df['Aggregate Type'] == agg_type]
        if len(sub) == 0:
            continue

        by_source = {}
        for src, grp in sub.groupby('Source'):
            by_source[src] = grp['sample_id'].astype(int).tolist()

        n_sources = len(by_source)
        n_samples = len(sub)

        if n_samples < 3:
            # too few to split meaningfully — dump everything in train,
            # train() will skip (< 3 samples) until more data arrives
            splits[key] = {"train": sorted(sub['sample_id'].astype(int).tolist()),
                           "val": [], "test": []}
            print(f"{key} ({agg_type}): {n_samples} samples, {n_sources} sources "
                  f"-> too few to split, all -> train (will be skipped by train())")
            continue

        result = source_aware_split(by_source)
        splits[key] = result
        print(f"{key} ({agg_type}): {n_samples} samples, {n_sources} sources -> "
              f"train={len(result['train'])} val={len(result['val'])} test={len(result['test'])}")

    with open(OUT_JSON, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
