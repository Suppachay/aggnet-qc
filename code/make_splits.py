"""
Generate splits.json for dataset3 — held-out test (10%) + 5-fold CV pool.

Structure: {model_key: {"test": [...], "folds": [[...], [...], [...], [...], [...]]}}

Groups samples by (Aggregate Type, Source) so a single source's samples are
distributed proportionally rather than dumped entirely into one split
(avoids leakage + avoids empty splits when a type has few dominant sources).

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
K_FOLDS    = 5
TEST_RATIO = 0.10

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


def split_test_and_pool(sample_ids_by_source, test_ratio=TEST_RATIO, seed=SEED):
    """Carve out a held-out test set proportionally within each source group."""
    rng = random.Random(seed)
    test_ids, pool_ids = [], []

    for src, ids in sorted(sample_ids_by_source.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)

        if n < 5:
            # too small to carve out a test slice without gutting the group
            pool_ids.extend(ids)
            continue

        n_test = max(1, round(n * test_ratio))
        test_ids.extend(ids[:n_test])
        pool_ids.extend(ids[n_test:])

    return sorted(test_ids), sorted(pool_ids)


def assign_folds(sample_ids_by_source_in_pool, k=K_FOLDS, seed=SEED):
    """Round-robin fold assignment within each source group (keeps each
    source spread across folds rather than concentrated in one).

    The starting fold index rotates per source group -- without this, every
    group's round-robin restarts at fold 0, so with many small groups fold 0
    systematically absorbs the most samples (observed: [91,46,33,29,19]
    instead of a roughly even split).
    """
    rng = random.Random(seed + 1)  # different seed stream than test split
    folds = [[] for _ in range(k)]
    offset = 0

    for src, ids in sorted(sample_ids_by_source_in_pool.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            folds[(i + offset) % k].append(sid)
        offset += 1

    for f in folds:
        f.sort()
    return folds


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

        if n_samples < K_FOLDS + 2:
            # too few to fold meaningfully — everything into fold 0, no test
            # train_kfold() will skip (mirrors old train() < 3 samples rule)
            all_ids = sorted(sub['sample_id'].astype(int).tolist())
            splits[key] = {"test": [], "folds": [all_ids] + [[] for _ in range(K_FOLDS - 1)]}
            print(f"{key} ({agg_type}): {n_samples} samples, {n_sources} sources "
                  f"-> too few to split, all -> fold 0 (will be skipped by train_kfold())")
            continue

        test_ids, pool_ids = split_test_and_pool(by_source)
        pool_by_source = {}
        for src, ids in by_source.items():
            pool_by_source[src] = [i for i in ids if i in set(pool_ids)]
        folds = assign_folds(pool_by_source)

        splits[key] = {"test": test_ids, "folds": folds}
        fold_sizes = [len(f) for f in folds]
        print(f"{key} ({agg_type}): {n_samples} samples, {n_sources} sources -> "
              f"test={len(test_ids)}  pool={len(pool_ids)}  folds={fold_sizes}")

    with open(OUT_JSON, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
