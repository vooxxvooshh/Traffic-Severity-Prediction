# TODO - Traffic Severity Prediction

## Part A: Improve Model Accuracy
- [ ] Verify feature name/mask consistency between training artifacts and inference (`feature_names_all`, `feature_mask`).
- [ ] Improve imbalance handling during model fitting (consistent `class_weight` / sample weights where applicable).
- [x] Improve hyperparameter search configuration (slightly broaden spaces / increase iterations).
- [x] Ensure evaluation artifacts (confusion matrix + per-class metrics) are saved to `model_comparison.csv` and `best_model_metrics.json`.


## Part B: Auth UI/Backend Completion Check
- [ ] Verify `/signup`, `/login`, `/logout` routes work end-to-end with existing frontend JS.
- [ ] Verify cookie/JWT auth protects `/dashboard` correctly.
- [ ] Verify `dashboard.html`/frontend redirect behavior when unauthenticated.

## After code edits
- [ ] Run `python main.py` to retrain and generate updated accuracy + F1 metrics.
- [ ] Run a quick API smoke test (start uvicorn) and verify `/health`.

