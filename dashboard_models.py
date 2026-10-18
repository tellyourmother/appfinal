"""Small-window evaluation for the legacy dashboard (not a future-game forecast)."""

import logging
from numbers import Integral

import lightgbm as lgb
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

MIN_WINDOW = 3
MAX_WINDOW = 10
STATS = ('PTS', 'REB', 'AST', 'MIN')
MODEL_NAMES = ('LightGBM', 'XGBoost')
COMPARISON_COLUMNS = ['Stat', 'True'] + [
    f'{model}_{field}'
    for model in MODEL_NAMES
    for field in ('Pred', 'MAE', 'RMSE', 'Status')
]
EVALUATION_NOTE = (
    'Evaluation uses only the latest held-out game. R² is unavailable because '
    'it requires at least two observations. MAE and RMSE both equal the absolute '
    'error for this one game; they do not establish forecast accuracy.'
)


def make_lightgbm(window_size):
    # The default 20 samples per leaf blocks splits in every supported window.
    # Small leaves/bins permit learning; shallow trees limit complexity, but do
    # not make a 3–10 game sample sufficient for reliable accuracy estimates.
    return lgb.LGBMRegressor(
        min_child_samples=1, min_data_in_bin=1,
        num_leaves=min(4, window_size), max_depth=2,
        random_state=42, n_jobs=1, verbosity=-1,
    )


def run_models(df, window_size=5, *, warn=None):
    """Return finite estimates/ranges and per-model statuses for one holdout.

    ``df`` must contain chronologically ordered, preprocessed game rows. The
    requested window is never silently shortened or filled with older rows.
    Unavailable statistics are omitted from the three result dictionaries.
    ``warn`` lets the dashboard surface validation/failure messages directly.
    """
    warn = warn if warn is not None else logging.getLogger(__name__).warning
    prediction, lower, upper, results = {}, {}, {}, []

    def output():
        return prediction, lower, upper, pd.DataFrame(results, columns=COMPARISON_COLUMNS)

    if (isinstance(window_size, bool) or not isinstance(window_size, Integral)
            or not MIN_WINDOW <= window_size <= MAX_WINDOW):
        warn(f'Training window must be an integer from {MIN_WINDOW} to {MAX_WINDOW} games.')
        return output()
    if len(df) < window_size + 1:
        warn(f'Not enough usable games: need {window_size} training games plus '
             f'1 held-out game ({window_size + 1} preprocessed rows); found {len(df)}. '
             'No model estimates are available.')
        return output()

    features = [col for col in df.columns if col.endswith('_prev')]
    if not features:
        warn('No previous-game features are available. No model estimates are available.')
        return output()
    recent = df.iloc[-(window_size + 1):]
    X = recent[features].apply(pd.to_numeric, errors='coerce').astype(float)
    if not np.isfinite(X.to_numpy()).all():
        warn('The selected training window or held-out game has missing or nonfinite '
             'features. No model estimates are available.')
        return output()
    X_train, X_test = X.iloc[:-1], X.iloc[-1:]
    if not X_train.nunique().gt(1).any():
        warn('Previous-game features do not vary within this training window. '
             'No model estimates are available; try a larger window.')
        return output()

    factories = {
        'LightGBM': lambda: make_lightgbm(window_size),
        'XGBoost': lambda: XGBRegressor(verbosity=0, random_state=42, n_jobs=1),
    }
    for stat in STATS:
        if stat not in recent:
            warn(f'{stat}: game data is missing this statistic; estimate unavailable.')
            continue
        y = pd.to_numeric(recent[stat], errors='coerce').astype(float)
        if not np.isfinite(y.to_numpy()).all():
            warn(f'{stat}: missing or nonfinite outcomes in the selected window; estimate unavailable.')
            continue
        y_train, actual = y.iloc[:-1], float(y.iloc[-1])
        constant_target = y_train.nunique() == 1
        if constant_target:
            warn(f'{stat}: all training outcomes are identical. A constant estimate '
                 'does not demonstrate predictive skill.')

        row = {'Stat': stat, 'True': round(actual, 2)}
        valid = []
        for name, factory in factories.items():
            # Missing values remain absent in the table and are never averaged.
            row.update({f'{name}_{field}': None for field in ('Pred', 'MAE', 'RMSE')})
            try:
                model = factory()
                model.fit(X_train, y_train)
                values = np.asarray(model.predict(X_test), dtype=float)
                if values.shape != (1,) or not np.isfinite(values).all():
                    raise ValueError('expected one finite prediction')
                estimate = float(values[0])
                error = abs(actual - estimate)
                if not np.isfinite(error):
                    raise ValueError('evaluation error is nonfinite')
            except Exception as exc:
                row[f'{name}_Status'] = f'Unavailable: {type(exc).__name__}: {exc}'
                warn(f'{name} could not evaluate {stat}: {exc}. Its output was excluded.')
                continue

            valid.append(estimate)
            row.update({
                f'{name}_Pred': round(estimate, 2),
                f'{name}_MAE': round(error, 2),
                f'{name}_RMSE': round(error, 2),
                f'{name}_Status': 'Constant training target' if constant_target else 'OK',
            })

        results.append(row)
        if not valid:
            warn(f'{stat}: neither model produced a valid estimate. Prediction unavailable.')
            continue
        if len(valid) == 1:
            warn(f'{stat}: only one model succeeded. Showing a single-model estimate; '
                 'no model range is available.')
        # Divide before summing to avoid overflow when combining large finite outputs.
        prediction[stat] = round(sum(value / len(valid) for value in valid), 2)
        lower[stat], upper[stat] = round(min(valid), 2), round(max(valid), 2)

    return output()
