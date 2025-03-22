import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats, LeagueGameFinder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time

st.set_page_config(layout="wide")
st.title("🏀 NBA Player Performance Dashboard")

@st.cache_data
def get_player_list():
    return players.get_active_players()

@st.cache_data
def get_team_map():
    return {t["abbreviation"]: t["full_name"] for t in teams.get_teams()}

@st.cache_data
def get_player_id(name):
    pl = get_player_list()
    match = [p for p in pl if name.lower() in p["full_name"].lower()]
    return match[0]["id"], match[0]["full_name"] if match else (None, None)

@st.cache_data
def get_game_log(player_id):
    return PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]

@st.cache_data
def get_team_strength():
    df = LeagueDashTeamStats(season='2024-25', season_type_all_star='Regular Season').get_data_frames()[0]
    return df[['TEAM_NAME', 'W_PCT', 'PTS', 'PLUS_MINUS']].rename(columns={
        'W_PCT': 'OPP_W_PCT', 'PTS': 'OPP_PTS', 'PLUS_MINUS': 'OPP_PLUS_MINUS'
    })

def preprocess(df, team_map, team_strength):
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')
    df['HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['OPP_ABBR'] = df['MATCHUP'].str.extract(r'@ (\w+)|vs\. (\w+)').bfill(axis=1).iloc[:, 0]
    df['OPPONENT'] = df['OPP_ABBR'].map(team_map)
    df = df.merge(team_strength, left_on='OPPONENT', right_on='TEAM_NAME', how='left')

    base_stats = ['MIN', 'PTS', 'REB', 'AST', 'FG3M', 'FGA', 'FG_PCT', 'FG3A', 'FG3_PCT',
                  'FTA', 'FT_PCT', 'STL', 'BLK', 'TOV', 'HOME',
                  'OPP_W_PCT', 'OPP_PTS', 'OPP_PLUS_MINUS']
    
    for col in base_stats:
        df[f'{col}_prev'] = df[col].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df

def weighted_lgb_model(df, target, quantile):
    features = [f for f in df.columns if '_prev' in f]
    df['weight'] = np.linspace(0.3, 1.0, len(df))
    model = lgb.LGBMRegressor(objective='quantile', alpha=quantile, n_estimators=100)
    model.fit(df[features], df[target], sample_weight=df['weight'])
    return model

def predict_stats(df):
    stats = ['PTS', 'REB', 'AST', 'MIN']
    pred, low, high = {}, {}, {}
    for stat in stats:
        m50 = weighted_lgb_model(df, stat, 0.5)
        m10 = weighted_lgb_model(df, stat, 0.1)
        m90 = weighted_lgb_model(df, stat, 0.9)
        X_pred = df[[f for f in df.columns if '_prev' in f]].iloc[[-1]]
        pred[stat] = m50.predict(X_pred)[0]
        low[stat] = m10.predict(X_pred)[0]
        high[stat] = m90.predict(X_pred)[0]
    return pred, low, high

def get_next_opponent(game_df):
    last_matchup = game_df['MATCHUP'].iloc[0]
    team_abbr = last_matchup.split(' ')[0]
    return get_team_map().get(team_abbr, team_abbr)

def plot_stat_trend(df, pred):
    stats = ['MIN', 'PTS', 'REB', 'AST']
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    df['Label'] = df['GAME_DATE'].dt.strftime('%b %d') + ' ' + df['MATCHUP']
    fig = make_subplots(rows=2, cols=2, subplot_titles=stats)
    for i, stat in enumerate(stats):
        row, col = i//2+1, i%2+1
        y_vals = list(df[stat]) + [pred[stat]]
        x_vals = list(df['Label']) + ['Predicted']
        colors = ['tomato' if v > df[stat].mean() else 'skyblue' for v in df[stat]] + ['gold']
        rolling = df[stat].rolling(5).mean()

        fig.add_trace(go.Bar(x=x_vals, y=y_vals, marker_color=colors, name=stat), row=row, col=col)
        fig.add_trace(go.Scatter(x=df['Label'], y=rolling, mode='lines',
                                 line=dict(color='black', dash='dash'), name='Rolling Avg'), row=row, col=col)
    fig.update_layout(height=800, title="📈 Last 15 Games + Predicted")
    st.plotly_chart(fig, use_container_width=True)

def team_bar_chart(df):
    st.subheader("📊 Average Stats vs Opponents")
    df = df.groupby('OPP_ABBR')[['PTS', 'REB', 'AST']].mean().sort_values('PTS', ascending=False)
    fig = px.bar(df, x=df.index, y=['PTS', 'REB', 'AST'], barmode='group')
    st.plotly_chart(fig, use_container_width=True)

def timeline_heatmap(df):
    st.subheader("🗓️ Heatmap: PTS vs Opponent by Month")
    df['Month'] = df['GAME_DATE'].dt.strftime('%b')
    pivot = df.pivot_table(index='OPP_ABBR', columns='Month', values='PTS', aggfunc='mean')
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap='coolwarm', annot=True, fmt=".1f", linewidths=0.5)
    st.pyplot(plt.gcf())

def full_team_heatmap(df):
    st.subheader("🗺️ Career Avg Stats vs Each Team")
    df['TEAM'] = df['MATCHUP'].str.extract(r'@ (\w+)|vs\. (\w+)').bfill(axis=1).iloc[:, 0]
    agg = df.groupby('TEAM')[['PTS', 'REB', 'AST', 'MIN']].mean().round(1)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(agg, annot=True, cmap='coolwarm', fmt=".1f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# === UI ===
players_list = sorted([p["full_name"] for p in get_player_list()])
player_name = st.selectbox("Search for a player", players_list, index=players_list.index("Jayson Tatum"))

if player_name:
    with st.spinner("Crunching data..."):
        player_id, full_name = get_player_id(player_name)
        gamelog = get_game_log(player_id)
        team_map = get_team_map()
        strength_df = get_team_strength()
        df = preprocess(gamelog.copy(), team_map, strength_df)
        pred, low, high = predict_stats(df)
        next_opp = get_next_opponent(gamelog)

        st.markdown(f"**Next Opponent (auto-detected):** `{next_opp}`")

        st.subheader("🔮 Predicted Next Game Stats (w/ Confidence Range)")
        cols = st.columns(4)
        for i, stat in enumerate(['MIN', 'PTS', 'REB', 'AST']):
            cols[i].metric(stat, f"{pred[stat]:.1f}", f"{low[stat]:.1f} – {high[stat]:.1f}")

        plot_stat_trend(df, pred)
        team_bar_chart(df)
        timeline_heatmap(df)
        full_team_heatmap(gamelog)
