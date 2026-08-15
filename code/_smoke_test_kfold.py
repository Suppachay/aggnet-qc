"""Quick local smoke test for train_kfold() — tiny epoch counts, just to
verify the fold loop / aggregation / final retrain / test-eval code path
runs end-to-end without crashing. NOT a real training run."""
import os
import aggnet_dataset3 as m

# point at local paths for this smoke test only (module hardcodes /workspace/AggNet/...)
m.DATA_DIR    = "data/dataset3"
m.LABELS_CSV  = os.path.join(m.DATA_DIR, "labels.csv")
m.SPLITS_JSON = os.path.join(m.DATA_DIR, "splits.json")
m.SAVE_DIR    = "outputs/_smoke_test"
m.MODEL_DIR   = "models/_smoke_test"
os.makedirs(m.SAVE_DIR, exist_ok=True)
os.makedirs(m.MODEL_DIR, exist_ok=True)

# shrink for a fast local smoke test (real run uses the normal values on Marimo)
m.FREEZE_EPOCHS = 2
m.MAX_EPOCHS = 3
m.PATIENCE = 2
m.K_FOLDS = 5

result = m.train_kfold(agg_filter='Aggregate 3_8inch', k=5)
print("\n\nSMOKE TEST RESULT:", result)
assert result is not None, "train_kfold returned None unexpectedly"
assert len(result['fold_metrics']) == 5, f"expected 5 fold results, got {len(result['fold_metrics'])}"
print("\nSMOKE TEST PASSED")
