"""Offline regression checks; run with python -m unittest discover -s tests."""

import importlib
import importlib.metadata
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import pandas as pd
from packaging.requirements import Requirement

import dashboard_models as models

ROOT = Path(__file__).resolve().parents[1]


def prepared_games(count=12):
    return pd.DataFrame({
        'PTS_prev': np.arange(count, dtype=float),
        'MIN_prev': np.arange(count, dtype=float) + 20,
        **{stat: np.arange(count, dtype=float) * 2 + 1 for stat in models.STATS},
    })


def estimator(value):
    model = Mock()
    model.predict.return_value = [value]
    return model


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.messages = []
        self.lgb = estimator(10)
        self.xgb = estimator(14)
        self.lgb_factory = self.enterContext(patch.object(models.lgb, 'LGBMRegressor', return_value=self.lgb))
        self.xgb_factory = self.enterContext(patch.object(models, 'XGBRegressor', return_value=self.xgb))

    def run_models(self, df=None, window=5):
        return models.run_models(prepared_games() if df is None else df, window, warn=self.messages.append)

    def test_all_supported_windows_use_exact_training_rows_and_separate_holdout(self):
        for window in range(3, 11):
            with self.subTest(window=window):
                df = prepared_games(window + 3)
                prediction, low, high, comparison = self.run_models(df, window)
                self.assertEqual(prediction, dict.fromkeys(models.STATS, 12))
                self.assertEqual(low, dict.fromkeys(models.STATS, 10))
                self.assertEqual(high, dict.fromkeys(models.STATS, 14))
                X, y = self.lgb.fit.call_args.args
                self.assertEqual(len(X), window)
                self.assertEqual(list(X.index), list(df.index[-window-1:-1]))
                self.assertEqual(list(y.index), list(X.index))
                self.assertEqual(list(self.lgb.predict.call_args.args[0].index), [df.index[-1]])
                self.assertEqual(comparison['True'].tolist(), [df['PTS'].iloc[-1]] * 4)
                self.assertLessEqual(2 * self.lgb_factory.call_args.kwargs['min_child_samples'], window)
                self.assertEqual(self.lgb_factory.call_args.kwargs['min_data_in_bin'], 1)

    def test_minimum_data_counts_preprocessed_rows(self):
        for window in range(3, 11):
            with self.subTest(window=window):
                self.assertFalse(self.run_models(prepared_games(window), window)[0])
                self.lgb_factory.assert_not_called()
                self.assertIn('plus 1 held-out game', self.messages[-1])
        self.assertTrue(self.run_models(prepared_games(4), 3)[0])

    def test_invalid_windows_do_not_train(self):
        for window in (0, 2, 11, -1, 3.5, True, '5', None):
            with self.subTest(window=window):
                self.assertFalse(self.run_models(window=window)[0])
        self.lgb_factory.assert_not_called()

    def test_successful_single_game_errors_have_no_r2_or_metric_warnings(self):
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            comparison = self.run_models(prepared_games(6))[3]
        self.assertEqual(emitted, [])
        self.assertFalse(any('R2' in col for col in comparison))
        self.assertEqual(comparison['LightGBM_MAE'].tolist(), [1] * 4)
        self.assertEqual(comparison['XGBoost_RMSE'].tolist(), [3] * 4)
        self.assertIn('at least two observations', models.EVALUATION_NOTE)
        self.assertIn('do not establish forecast accuracy', models.EVALUATION_NOTE)

    def test_either_estimator_can_fail_without_contaminating_fallback(self):
        for broken, survivor in ((self.lgb, 14), (self.xgb, 10)):
            with self.subTest(survivor=survivor):
                broken.fit.side_effect = RuntimeError('mock fit failure')
                prediction, low, high, comparison = self.run_models()
                self.assertEqual(prediction, dict.fromkeys(models.STATS, survivor))
                self.assertEqual(prediction, low)
                self.assertEqual(low, high)
                self.assertTrue(any('only one model succeeded' in m for m in self.messages))
                failed_name = 'LightGBM' if broken is self.lgb else 'XGBoost'
                self.assertTrue(comparison[f'{failed_name}_Pred'].isna().all())
                self.assertTrue(comparison[f'{failed_name}_Status'].str.contains('mock fit failure').all())
                broken.fit.side_effect = None

    def test_both_fail_no_prediction_or_range_and_no_nanmean_warning(self):
        self.lgb.fit.side_effect = ValueError('broken LGB')
        self.xgb.predict.side_effect = ValueError('broken XGB')
        with warnings.catch_warnings(record=True) as emitted:
            result = self.run_models()
        self.assertEqual(result[:3], ({}, {}, {}))
        self.assertEqual(len(result[3]), 4)
        self.assertEqual(emitted, [])
        self.assertTrue(any('neither model' in m for m in self.messages))

    def test_nonfinite_and_malformed_outputs_are_excluded(self):
        for output in ([np.nan], [np.inf], [-np.inf], [], [1, 2], [[10]], ['bad']):
            with self.subTest(output=output):
                self.lgb.predict.return_value = output
                self.assertEqual(self.run_models()[0], dict.fromkeys(models.STATS, 14))
        self.xgb.predict.return_value = [np.nan]
        self.assertEqual(self.run_models()[:3], ({}, {}, {}))

    def test_constructor_failure_is_reported(self):
        self.lgb_factory.side_effect = RuntimeError('constructor failed')
        self.assertEqual(self.run_models()[0]['PTS'], 14)
        self.assertTrue(any('constructor failed' in m for m in self.messages))

    def test_missing_nonfinite_and_constant_features_do_not_train(self):
        cases = [pd.DataFrame(), prepared_games().drop(columns=['PTS_prev', 'MIN_prev'])]
        for value in (np.nan, np.inf, 'bad'):
            df = prepared_games().astype({'PTS_prev': object})
            df.loc[df.index[-1], 'PTS_prev'] = value
            cases.append(df)
        constant = prepared_games()
        constant[['PTS_prev', 'MIN_prev']] = 1
        cases.append(constant)
        for df in cases:
            with self.subTest(shape=df.shape):
                self.assertEqual(self.run_models(df)[:3], ({}, {}, {}))
        self.lgb_factory.assert_not_called()

    def test_invalid_target_only_withholds_affected_statistic(self):
        for value in (np.nan, np.inf):
            df = prepared_games()
            df.loc[df.index[-1], 'PTS'] = value
            result = self.run_models(df)
            self.assertNotIn('PTS', result[0])
            self.assertEqual(set(result[0]), {'MIN', 'REB', 'AST'})
        self.assertNotIn('PTS', self.run_models(prepared_games().drop(columns='PTS'))[0])

    def test_constant_target_is_explicitly_reported(self):
        df = prepared_games()
        df['PTS'] = 5
        comparison = self.run_models(df)[3].set_index('Stat')
        self.assertEqual(comparison.loc['PTS', 'LightGBM_Status'], 'Constant training target')
        self.assertTrue(any('does not demonstrate predictive skill' in m for m in self.messages))

    def test_old_invalid_rows_do_not_change_selected_window(self):
        df = prepared_games()
        df.loc[0, 'PTS_prev'] = np.nan
        self.assertTrue(self.run_models(df)[0])


class InstalledDependencyTests(unittest.TestCase):
    def test_manifest_dependencies_are_installed_and_importable(self):
        import_names = {'scikit-learn': 'sklearn', 'nba-api': 'nba_api'}
        requirements = [Requirement(line) for line in (ROOT / 'requirements.txt').read_text().splitlines()
                        if line.strip() and not line.startswith('#')]
        self.assertIn('lightgbm', [req.name for req in requirements])
        for requirement in requirements:
            with self.subTest(dependency=requirement.name):
                self.assertIn(importlib.metadata.version(requirement.name), requirement.specifier)
                importlib.import_module(import_names.get(requirement.name, requirement.name))

    def test_real_lightgbm_can_split_at_every_supported_window(self):
        for window in range(3, 11):
            with self.subTest(window=window):
                df = prepared_games(window)
                X, y = df[['PTS_prev', 'MIN_prev']], df['PTS']
                model = models.make_lightgbm(window)
                model.fit(X, y)
                self.assertGreater(np.ptp(model.predict(X)), 0)
                self.assertTrue(any(t['num_leaves'] > 1 for t in model.booster_.dump_model()['tree_info']))

    def test_real_estimators_work_together_on_small_windows(self):
        for window in (3, 5, 10):
            with self.subTest(window=window):
                messages = []
                prediction, low, high, comparison = models.run_models(
                    prepared_games(window + 1), window, warn=messages.append)
                self.assertEqual(messages, [])
                self.assertEqual(set(prediction), set(models.STATS))
                for name in models.MODEL_NAMES:
                    self.assertTrue(comparison[f'{name}_Status'].eq('OK').all())
                for stat in models.STATS:
                    self.assertTrue(np.isfinite(prediction[stat]))
                    self.assertLessEqual(low[stat], prediction[stat])
                    self.assertLessEqual(prediction[stat], high[stat])


if __name__ == '__main__':
    unittest.main()
