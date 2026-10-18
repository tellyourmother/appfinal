# Dashboard installation and small-window reliability

This change is based directly on `main` at `b55297af592a40162a476338ffc8b81d68b9b821`.
It is independent of Stage 1 foundation PR #1. The season selection, opponent
selection, feature definitions, and latest-game holdout workflow are unchanged.

## Behavior

- `requirements.txt` includes `lightgbm==4.6.0`. The existing XGBoost pin remains
  `1.7.5`; no new test-only dependency is required.
- Model evaluation lives in `dashboard_models.py`, so it can be tested without
  starting Streamlit or requesting NBA data.
- Supported windows remain 3–10 games, default five. A window of N requires N+1
  usable, preprocessed rows: N for training and one held-out observation. Creating
  lag features consumes another raw row, so a complete raw log needs at least N+2
  games. Missing values can reduce the usable count further.
- LightGBM uses `min_child_samples=1`, `min_data_in_bin=1`, depth 2 and at most
  four leaves. This permits splits in tiny windows; it does not establish
  predictive quality. Both estimators use a fixed random seed and one CPU worker.
- Invalid windows, missing/nonfinite features, and nonvarying training features
  withhold estimates with an explanation. Invalid outcomes withhold the affected
  statistic. Identical training outcomes receive an explicit constant-target warning.
- Construction, fitting, and prediction failures are reported in the interface.
  Malformed and nonfinite predictions are excluded from the mean and range.
  One successful estimator is labeled as a single-model result with no displayed
  range; two failures produce an unavailable value. Other statistics and history
  charts remain usable.
- R² is neither computed nor plotted for a one-game holdout. The interface explains
  why it is undefined and why one-game MAE/RMSE both equal absolute error. The
  metric chart now correctly identifies model and metric names.
- All six existing tabs and their visualization types remain. Trend charts omit
  unavailable estimates; empty history stops empty plots cleanly. Labels identify
  held-out estimates and model spread rather than a future forecast or confidence
  interval.

## Reproduce the offline checks

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -p 'test_dashboard_*.py' -v
.venv/bin/python -m compileall -q appfinalv5webapp.py dashboard_models.py tests
```

The scoped test pattern also works after merging the separate Stage 1 tests.

Validation performed on Linux x86_64 with Python 3.12.14:

- All 22 tests passed, including subtests for every supported window (3–10).
- Every manifest dependency imported and satisfied its declared version specifier.
- Real LightGBM generated varying predictions and actual tree splits at every
  supported window. Both real estimators produced finite results together at
  windows 3, 5, and 10.
- Mocked outputs cover successful averaging, either estimator failing, both
  failing, constructor/predict failures, NaN/infinite/malformed output, incomplete
  windows, invalid features/targets, and constant targets.
- Streamlit AppTest used mocked NBA logs and blocked HTTP requests. It verified
  all six tabs, chart model labels, lag-adjusted minimum history, empty history,
  single-model fallbacks, complete failure, and partially unavailable statistics.
- Compilation and whitespace checks passed.

Installed versions included LightGBM 4.6.0, XGBoost 1.7.5, Streamlit 1.66.0,
NumPy 2.5.4, pandas 3.0.6, and scikit-learn 1.9.1. These record the tested
environment; the previously unpinned packages remain unpinned.

`pip check` reports `xgboost 1.7.5 is not supported on this platform`: the installed
wheel's internal metadata declares CPython 3.10 tags despite its generic Python 3
filename. Installation, import, and real estimator checks succeeded on Python
3.12.14. This pre-existing XGBoost packaging issue remains; this change does not
upgrade XGBoost or rewrite package metadata.

No live NBA availability or forecasting accuracy was measured. The opponent
selector still does not feed a next-game forecast, and the existing season data
selection is unchanged. Small windows can overfit even when training succeeds.

## Parameter and metric references

- [LightGBM 4.6.0 LGBMRegressor](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMRegressor.html)
- [LightGBM 4.6.0 parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html)
- [scikit-learn R² documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
