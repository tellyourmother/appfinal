"""Exercise the full Streamlit script with offline NBA logs and model outputs."""

import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

import dashboard_models as models

APP = Path(__file__).resolve().parents[1] / 'appfinalv5webapp.py'


def game_log(count):
    stats = ('MIN', 'PTS', 'REB', 'AST', 'FG3M', 'FGA', 'FG_PCT', 'FG3A',
             'FG3_PCT', 'FTA', 'FT_PCT', 'STL', 'BLK', 'TOV')
    return pd.DataFrame({
        'GAME_DATE': pd.date_range('2025-01-01', periods=count),
        'MATCHUP': ['BOS vs. LAL' if i % 2 else 'BOS @ LAL' for i in range(count)],
        **{stat: np.arange(count, dtype=float) + 1 for stat in stats},
    }).iloc[::-1]


class DashboardUITests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        self.addCleanup(st.cache_data.clear)
        self.enterContext(patch('requests.sessions.Session.request', side_effect=AssertionError('Unexpected network call')))
        self.enterContext(patch('nba_api.stats.static.players.get_active_players',
                                return_value=[{'id': 1, 'full_name': 'Jayson Tatum'}]))
        self.enterContext(patch('nba_api.stats.static.teams.get_teams', return_value=[
            {'abbreviation': 'LAL', 'full_name': 'Los Angeles Lakers'},
            {'abbreviation': 'BOS', 'full_name': 'Boston Celtics'},
        ]))
        self.logs = self.enterContext(patch('nba_api.stats.endpoints.PlayerGameLog'))
        self.logs.return_value.get_data_frames.return_value = [game_log(12)]
        strength = self.enterContext(patch('nba_api.stats.endpoints.LeagueDashTeamStats'))
        strength.return_value.get_data_frames.return_value = [pd.DataFrame({
            'TEAM_NAME': ['Los Angeles Lakers', 'Boston Celtics'], 'W_PCT': [0.5, 0.7],
        })]
        self.lgb, self.xgb = Mock(), Mock()
        self.lgb.predict.return_value = [10.0]
        self.xgb.predict.return_value = [14.0]
        self.enterContext(patch.object(models.lgb, 'LGBMRegressor', return_value=self.lgb))
        self.enterContext(patch.object(models, 'XGBRegressor', return_value=self.xgb))

    def app(self):
        result = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertEqual(len(result.exception), 0, [e.message for e in result.exception])
        self.assertEqual(len(result.tabs), 6)
        self.assertTrue(any('R² is unavailable' in info.value for info in result.info))
        return result

    def test_success_keeps_all_visualizations_and_parses_metric_chart_labels(self):
        app = self.app()
        self.assertEqual([m.value for m in app.metric], ['12.0'] * 4)
        charts = app.get('plotly_chart')
        self.assertEqual(len(charts), 5)
        chart = json.loads(charts[0].proto.spec)
        self.assertEqual({trace['name'] for trace in chart['data']}, {'LightGBM', 'XGBoost'})
        self.assertNotIn('R2', charts[0].proto.spec)
        self.assertEqual(app.slider[0].min, 3)
        self.assertEqual(app.slider[0].max, 10)
        self.assertEqual(app.slider[0].value, 5)
        self.assertEqual(len(self.lgb.fit.call_args.args[0]), 5)

    def test_minimum_raw_logs_after_lagging(self):
        # Five raw logs leave three training rows and one holdout after lagging.
        self.logs.return_value.get_data_frames.return_value = [game_log(5)]
        app = self.app()
        self.assertTrue(all(m.value == 'Unavailable' for m in app.metric))
        app.slider[0].set_value(3).run()
        self.assertFalse(app.exception)
        self.assertTrue(all(m.value == '12.0' for m in app.metric))
        self.assertEqual(len(self.lgb.fit.call_args.args[0]), 3)

    def test_insufficient_history_preserves_historical_charts(self):
        self.logs.return_value.get_data_frames.return_value = [game_log(4)]
        app = self.app()
        self.assertTrue(all(m.value == 'Unavailable' for m in app.metric))
        self.assertTrue(any('Not enough usable games' in w.value for w in app.warning))
        self.assertEqual(len(app.get('plotly_chart')), 4)
        self.lgb.fit.assert_not_called()

    def test_no_usable_history_stops_empty_visualizations_cleanly(self):
        for count in (0, 1):
            with self.subTest(count=count):
                st.cache_data.clear()
                self.logs.return_value.get_data_frames.return_value = [game_log(count)]
                app = self.app()
                self.assertTrue(all(m.value == 'Unavailable' for m in app.metric))
                self.assertTrue(any('No usable game history' in i.value for i in app.info))

    def test_one_estimator_failure_explains_fallback(self):
        self.lgb.fit.side_effect = RuntimeError('mock unavailable model')
        app = self.app()
        self.assertTrue(all(m.value == '14.0' for m in app.metric))
        self.assertTrue(all(m.delta == 'Single model; range unavailable' for m in app.metric))
        self.assertTrue(any('mock unavailable model' in w.value for w in app.warning))

    def test_all_failures_leave_history_and_show_no_nan_estimates(self):
        self.lgb.fit.side_effect = RuntimeError('mock unavailable model')
        self.xgb.predict.return_value = [np.nan]
        app = self.app()
        self.assertTrue(all(m.value == 'Unavailable' for m in app.metric))
        self.assertTrue(any('neither model produced' in w.value for w in app.warning))
        self.assertEqual(len(app.get('plotly_chart')), 4)

    def test_partial_stat_failure_does_not_break_trends(self):
        self.lgb.predict.side_effect = [[np.nan], [10], [10], [10]]
        self.xgb.predict.side_effect = [[np.inf], [14], [14], [14]]
        app = self.app()
        self.assertEqual([m.value for m in app.metric], ['12.0', 'Unavailable', '12.0', '12.0'])
        trends = json.loads(app.get('plotly_chart')[1].proto.spec)
        pts = next(t for t in trends['data'] if t['type'] == 'bar' and t['name'] == 'PTS')
        self.assertNotIn('Held-out estimate', pts['x'])


if __name__ == '__main__':
    unittest.main()
