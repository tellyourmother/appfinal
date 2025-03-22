import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("🏀 NBA Player Stat Predictor")

# ========== Data Utils ==========
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

def get_player_id(name):
    result = [p for p in get_player_options() if name.lower() in p["full_name"].lower()]
    return (result[0]['id'], result[0]['full_name']) if result else (None, None)

@st.cache_data
def load_game_log(player_id):
    return PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]

def preprocess_games(df, team_strength_df, team_map):
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')
    df['HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['OPP_ABBR'] = df['MATCHUP'].str.extract(r'@ (\w+)|vs\. (\w+)').bfill(axis=1).iloc[:, 0]
    df['OPPONENT'] = df['OPP_ABBR'].map(team_map)

    df = df.merge(team_strength_df, left_on='OPPONENT', right_on='TEAM_NAME', how='left')
    df['W_PCT'] = df['W_PCT'].fillna(0.5)

    features = ['MIN', 'PTS', 'REB', 'AST', 'FG3M', 'FGA', 'FG_PCT', 'FG3A',
                'FG3_PCT', 'FTA', 'FT_PCT', 'STL', 'BLK', 'TOV', 'HOME']
    for col in features:
        df[f'{col}_prev'] = df[col].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df

# ========== Modeling ==========
def weighted_train(df, target, quantile):
    features = [col for col in df.columns if '_prev' in col]
    df['weight'] = np.linspace(0.3, 1, len(df))
    model = lgb.LGBMRegressor(objective='quantile', alpha=quantile, n_estimators=100)
    model.fit(df[features], df[target], sample_weight=df['weight'])
    return model

def predict_with_intervals(df):
    stats = ['PTS', 'REB', 'AST', 'MIN']
    preds, lowers, uppers = {}, {}, {}
    for stat in stats:
        model_mid = weighted_train(df, stat, 0.5)
        model_low = weighted_train(df, stat, 0.1)
        model_high = weighted_train(df, stat, 0.9)
        X_pred = df[[col for col in df.columns if '_prev' in col]].iloc[[-1]]
        preds[stat] = model_mid.predict(X_pred)[0]
        lowers[stat] = model_low.predict(X_pred)[0]
        uppers[stat] = model_high.predict(X_pred)[0]
    return preds, lowers, uppers

# ========== Visuals ==========
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
        fig.add_trace(go.Scatter(x=x_vals[:-1], y=df[stat].rolling(5).mean(), mode='lines', name='Rolling Avg',
                                 line=dict(color='black', dash='dash')), row=row, col=col)
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

    st.subheader("📊 Performance vs Team Tiers (Based on W%)")
    fig = px.bar(tier_avg, x='TIER', y=['PTS', 'REB', 'AST'], barmode='group',
                 color_discrete_map={"PTS": "gold", "REB": "skyblue", "AST": "lightgreen"})
    st.plotly_chart(fig, use_container_width=True)

# ========== UI ==========
player_list = sorted([p["full_name"] for p in get_player_options()])
player_name = st.selectbox("Search for a player", player_list, index=player_list.index("Jayson Tatum"))

if player_name:
    player_id, full_name = get_player_id(player_name)
    gamelog = load_game_log(player_id)
    team_map = get_team_abbr_map()
    team_strength = get_team_strength()
    df = preprocess_games(gamelog, team_strength, team_map)
    prediction, lower, upper = predict_with_intervals(df)

    team_abbr_list = sorted(team_map.keys())
    next_opponent = st.selectbox("Select next opponent team", team_abbr_list)
    st.markdown(f"**Next Opponent:** `{next_opponent}`")

    # === TABS ===
    tabs = st.tabs(["📊 Prediction", "📈 Trends", "🆚 Stats vs Teams", "🗺️ Heatmaps", "🏆 Tiers"])

    with tabs[0]:
        st.subheader("🔮 Predicted Stats (with 10th–90th Percentile Range)")
        cols = st.columns(4)
        for i, stat in enumerate(['MIN', 'PTS', 'REB', 'AST']):
            val = prediction[stat]
            low, high = lower[stat], upper[stat]
            cols[i].metric(stat, f"{val:.1f}", f"{low:.1f} – {high:.1f}")

    with tabs[1]:
        plot_stat_trend(df, prediction)

    with tabs[2]:
        stat_vs_team_bar(df)

    with tabs[3]:
        timeline_heatmap(df)

    with tabs[4]:
        tier_breakdown_chart(df)
