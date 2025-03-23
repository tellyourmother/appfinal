import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats
import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ========== Streamlit Config ==========
st.set_page_config(layout="wide")
st.title("🏀 NBA Player Performance Dashboard")

# ========== CACHED FUNCTIONS ==========
@st.cache_data
def get_player_options():
    return players.get_active_players()

@st.cache_data
def get_team_abbr_map():
    return {team['abbreviation']: team['full_name'] for team in teams.get_teams()}

@st.cache_data
def get_team_strength():
    df = LeagueDashTeamStats(season='2024-25', season_type_all_star='Regular Season').get_data_frames()[0]
    return df[['TEAM_NAME', 'W_PCT']]

@st.cache_data
def get_player_id(name):
    result = [p for p in get_player_options() if name.lower() in p["full_name"].lower()]
    return (result[0]['id'], result[0]['full_name']) if result else (None, None)

@st.cache_data
def load_game_log(player_id):
    return PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]

# ========== DATA PREP ==========
def preprocess_games(df, team_strength_df, team_map):
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')
    df['HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['OPP_ABBR'] = df['MATCHUP'].str.extract(r'@ (\w+)|vs\. (\w+)').bfill(axis=1).iloc[:, 0]
    df['OPPONENT'] = df['OPP_ABBR'].map(team_map)
    df = df.merge(team_strength_df, left_on='OPPONENT', right_on='TEAM_NAME', how='left')
    df['W_PCT'] = df['W_PCT'].fillna(0.5)

    base_stats = ['MIN', 'PTS', 'REB', 'AST', 'FG3M', 'FGA', 'FG_PCT', 'FG3A',
                  'FG3_PCT', 'FTA', 'FT_PCT', 'STL', 'BLK', 'TOV', 'HOME']
    for col in base_stats:
        df[f'{col}_prev'] = df[col].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df

# ========== MODEL: LightGBM vs XGBoost ==========
def run_models(df):
    stats = ['PTS', 'REB', 'AST', 'MIN']
    prediction = {}
    lower = {}
    upper = {}
    features = [col for col in df.columns if '_prev' in col]
    X = df[features]
    results = []

    for stat in stats:
        y = df[stat]
        X_train, X_test = X.iloc[:-1], X.iloc[-1:]
        y_train, y_test = y.iloc[:-1], y.iloc[-1:]

        try:
            lgbm = lgb.LGBMRegressor()
            lgbm.fit(X_train, y_train)
            lgb_train_preds = lgbm.predict(X_train)
            lgb_test_pred = lgbm.predict(X_test)[0]
            lgb_mae = mean_absolute_error(y_train, lgb_train_preds)
            lgb_r2 = r2_score(y_train, lgb_train_preds)
            lgb_rmse = np.sqrt(mean_squared_error(y_train, lgb_train_preds))
        except Exception as e:
            lgb_test_pred, lgb_mae, lgb_r2, lgb_rmse = np.nan, np.nan, np.nan, np.nan
            print(f"LightGBM error for {stat}: {e}")

        try:
            xgb = XGBRegressor(verbosity=0)
            xgb.fit(X_train, y_train)
            xgb_train_preds = xgb.predict(X_train)
            xgb_test_pred = xgb.predict(X_test)[0]
            xgb_mae = mean_absolute_error(y_train, xgb_train_preds)
            xgb_r2 = r2_score(y_train, xgb_train_preds)
            xgb_rmse = np.sqrt(mean_squared_error(y_train, xgb_train_preds))
        except Exception as e:
            xgb_test_pred, xgb_mae, xgb_r2, xgb_rmse = np.nan, np.nan, np.nan, np.nan
            print(f"XGBoost error for {stat}: {e}")

        ensemble_pred = np.nanmean([lgb_test_pred, xgb_test_pred])
        prediction[stat] = round(ensemble_pred, 2)
        lower[stat] = round(min(lgb_test_pred, xgb_test_pred), 2)
        upper[stat] = round(max(lgb_test_pred, xgb_test_pred), 2)

        results.append({
            'Stat': stat,
            'True': round(y_test.values[0], 2),
            'LightGBM_Pred': round(lgb_test_pred, 2),
            'XGBoost_Pred': round(xgb_test_pred, 2),
            'LightGBM_MAE': round(lgb_mae, 2),
            'XGBoost_MAE': round(xgb_mae, 2),
            'LightGBM_R2': round(lgb_r2, 2),
            'XGBoost_R2': round(xgb_r2, 2),
            'LightGBM_RMSE': round(lgb_rmse, 2),
            'XGBoost_RMSE': round(xgb_rmse, 2),
        })

    compare_df = pd.DataFrame(results)
    return prediction, lower, upper, compare_df

# ========== VISUALIZATIONS ==========
def plot_stat_trend(df, prediction):
    fig = make_subplots(rows=2, cols=2, subplot_titles=['MIN', 'PTS', 'REB', 'AST'])
    stat_map = ['MIN', 'PTS', 'REB', 'AST']
    df = df.copy().sort_values('GAME_DATE').reset_index(drop=True)
    df['Label'] = df['GAME_DATE'].dt.strftime('%b %d') + ' ' + df['MATCHUP']
    for i, stat in enumerate(stat_map):
        row, col = i//2 + 1, i%2 + 1
        y_vals = list(df[stat]) + [prediction[stat]]
        x_vals = list(df['Label']) + ['Predicted']
        avg_val = df[stat].mean()
        colors = ['tomato' if val > avg_val else 'skyblue' for val in df[stat]] + ['gold']
        fig.add_trace(go.Bar(x=x_vals, y=y_vals, marker_color=colors, name=stat), row=row, col=col)
        fig.add_trace(go.Scatter(x=x_vals[:-1], y=df[stat].rolling(5).mean(), mode='lines',
                                 line=dict(color='black', dash='dash'), name='Rolling Avg'), row=row, col=col)
    fig.update_layout(height=800, title="📈 Last 15 Games + Prediction")
    st.plotly_chart(fig, use_container_width=True)

def stat_vs_team_bar(df):
    df = df.groupby('OPP_ABBR')[['PTS', 'REB', 'AST']].mean().sort_values('PTS', ascending=False)
    st.subheader("📊 Avg Stats vs Each Team")
    fig = px.bar(df, x=df.index, y=['PTS', 'REB', 'AST'], barmode='group')
    st.plotly_chart(fig, use_container_width=True)

def timeline_heatmap(df):
    df['Month'] = df['GAME_DATE'].dt.strftime('%b')
    pivot = df.pivot_table(index='OPP_ABBR', columns='Month', values='PTS', aggfunc='mean')
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap='coolwarm', annot=True, fmt=".1f", linewidths=0.5)
    plt.title("🗓️ PTS Heatmap vs Team by Month")
    st.pyplot(plt.gcf())

def tier_breakdown_chart(df):
    sorted_teams = df[['OPPONENT', 'W_PCT']].drop_duplicates().sort_values('W_PCT', ascending=False)
    top10 = sorted_teams.head(10)['OPPONENT']
    bottom10 = sorted_teams.tail(10)['OPPONENT']

    def assign_tier(team):
        if team in top10.values:
            return "Top 10"
        elif team in bottom10.values:
            return "Bottom 10"
        else:
            return "Middle 10"

    df['TIER'] = df['OPPONENT'].apply(assign_tier)
    tier_avg = df.groupby('TIER')[['PTS', 'REB', 'AST']].mean().reset_index()

    st.subheader("🏆 Performance vs Team Tiers")
    fig = px.bar(tier_avg, x='TIER', y=['PTS', 'REB', 'AST'], barmode='group',
                 color_discrete_map={"PTS": "gold", "REB": "skyblue", "AST": "lightgreen"})
    st.plotly_chart(fig, use_container_width=True)

def show_stat_distribution(df):
    st.subheader("📈 Distribution of Key Stats")
    fig = px.box(df, y=['PTS', 'REB', 'AST', 'MIN'], points='all')
    st.plotly_chart(fig, use_container_width=True)

# ========== STREAMLIT UI ==========
player_list = sorted([p["full_name"] for p in get_player_options()])
player_name = st.selectbox("Search for a player", player_list, index=player_list.index("Jayson Tatum"))

if player_name:
    player_id, full_name = get_player_id(player_name)
    gamelog = load_game_log(player_id)
    team_map = get_team_abbr_map()
    team_strength = get_team_strength()
    df = preprocess_games(gamelog, team_strength, team_map)
    prediction, lower, upper, compare_df = run_models(df)

    team_abbr_list = sorted(team_map.keys())
    next_opponent = st.selectbox("Select next opponent team", team_abbr_list)
    st.markdown(f"**Next Opponent:** `{next_opponent}`")

    tabs = st.tabs([
        "📊 Prediction", "📈 Trends", "📊 Distribution", "🆚 Stats vs Teams", 
        "🗺️ Heatmaps", "🏆 Tiers"
    ])

    with tabs[0]:
        st.subheader("🔮 Ensemble Prediction (LGB + XGB)")
        cols = st.columns(4)
        for i, stat in enumerate(['MIN', 'PTS', 'REB', 'AST']):
            cols[i].metric(stat, f"{prediction[stat]:.1f}", f"{lower[stat]:.1f} – {upper[stat]:.1f}")

        with st.expander("📊 Compare LightGBM vs XGBoost"):
            st.dataframe(compare_df)
            metric_melt = compare_df.melt(id_vars='Stat', value_vars=[
                'LightGBM_MAE', 'XGBoost_MAE',
                'LightGBM_R2', 'XGBoost_R2',
                'LightGBM_RMSE', 'XGBoost_RMSE'
            ], var_name='Metric_Model', value_name='Value')
            metric_melt[['Metric', 'Model']] = metric_melt['Metric_Model'].str.extract(r'(.*)_(LightGBM|XGBoost)')
            fig = px.bar(metric_melt, x='Stat', y='Value', color='Model', facet_row='Metric',
                         barmode='group', height=600, title="📉 Model Metric Comparison")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        plot_stat_trend(df, prediction)

    with tabs[2]:
        show_stat_distribution(df)

    with tabs[3]:
        stat_vs_team_bar(df)

    with tabs[4]:
        timeline_heatmap(df)

    with tabs[5]:
        tier_breakdown_chart(df)
