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
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("🏀 NBA Player Performance Dashboard")

# ========== CACHING ==========
@st.cache_data
def get_player_options():
    return players.get_active_players()

@st.cache_data
def get_team_abbr_map():
    return {team['abbreviation']: team['full_name'] for team in teams.get_teams()}

@st.cache_data
def get_team_strength_full():
    return LeagueDashTeamStats(season='2024-25', season_type_all_star='Regular Season').get_data_frames()[0]

@st.cache_data
def get_player_id(name):
    result = [p for p in get_player_options() if name.lower() in p["full_name"].lower()]
    return (result[0]['id'], result[0]['full_name']) if result else (None, None)

@st.cache_data
def load_game_log(player_id):
    return PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]

# ========== PREPROCESS ==========
def preprocess_games(df, full_strength_df, team_map):
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')
    df['HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['OPP_ABBR'] = df['MATCHUP'].str.extract(r'@ (\w+)|vs\. (\w+)').bfill(axis=1).iloc[:, 0]
    df['OPPONENT'] = df['OPP_ABBR'].map(team_map)

    df = df.merge(full_strength_df[['TEAM_NAME', 'W_PCT', 'DEF_RATING', 'PACE']], 
                  left_on='OPPONENT', right_on='TEAM_NAME', how='left')
    df['W_PCT'] = df['W_PCT'].fillna(0.5)

    features = ['MIN', 'PTS', 'REB', 'AST', 'FG3M', 'FGA', 'FG_PCT', 'FG3A',
                'FG3_PCT', 'FTA', 'FT_PCT', 'STL', 'BLK', 'TOV', 'HOME']
    for col in features:
        df[f'{col}_prev'] = df[col].shift(1)
    return df.dropna().reset_index(drop=True)

# ========== PREDICTION ==========
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

# ========== DIFFICULTY CATEGORY ==========
def categorize_difficulty(def_rating, pace, w_pct):
    if def_rating < 110 and pace < 98 and w_pct > 0.6:
        return "🟥 Hard"
    elif def_rating < 113 or pace < 101 or w_pct > 0.5:
        return "🟨 Medium"
    else:
        return "🟩 Easy"

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

def plot_matchup_difficulty(df, opp_name):
    row = df[df['TEAM_NAME'] == opp_name].iloc[0]
    difficulty = categorize_difficulty(row['DEF_RATING'], row['PACE'], row['W_PCT'])
    st.subheader(f"🎯 Matchup Difficulty for {opp_name}")
    st.markdown(f"- **DEF_RATING:** {row['DEF_RATING']:.1f}")
    st.markdown(f"- **PACE:** {row['PACE']:.1f}")
    st.markdown(f"- **W_PCT:** {row['W_PCT']:.2f}")
    st.markdown(f"### Result: **{difficulty}**")

def plot_archetype_stats(df):
    st.subheader("📊 Performance vs Opponent Archetypes")

    df['PACE_TIER'] = pd.qcut(df['PACE'], q=3, labels=['Slow', 'Medium', 'Fast'])
    pace_avg = df.groupby('PACE_TIER')[['PTS', 'REB', 'AST']].mean().reset_index()
    st.markdown("#### 🏃‍♂️ By Opponent Pace")
    st.plotly_chart(px.bar(pace_avg, x='PACE_TIER', y=['PTS', 'REB', 'AST'], barmode='group'), use_container_width=True)

    df['DEF_TIER'] = pd.qcut(df['DEF_RATING'], q=3, labels=['Strong D', 'Average D', 'Weak D'])
    def_avg = df.groupby('DEF_TIER')[['PTS', 'REB', 'AST']].mean().reset_index()
    st.markdown("#### 🧱 By Opponent Defense")
    st.plotly_chart(px.bar(def_avg, x='DEF_TIER', y=['PTS', 'REB', 'AST'], barmode='group'), use_container_width=True)

# ========== UI ==========
players_list = sorted([p['full_name'] for p in get_player_options()])
player_name = st.selectbox("Choose a player", players_list, index=players_list.index("Jayson Tatum"))

if player_name:
    player_id, full_name = get_player_id(player_name)
    game_log = load_game_log(player_id)
    team_map = get_team_abbr_map()
    full_team_stats = get_team_strength_full()
    df = preprocess_games(game_log, full_team_stats, team_map)
    prediction, lower, upper = predict_with_intervals(df)

    next_opp = st.selectbox("Select next opponent", sorted(team_map.keys()))
    opp_name = team_map.get(next_opp, next_opp)

    tabs = st.tabs([
        "📊 Prediction", "📈 Trends", "🧠 Matchup Intelligence"
    ])

    with tabs[0]:
        st.subheader("🔮 Prediction (w/ Confidence Intervals)")
        cols = st.columns(4)
        for i, stat in enumerate(['MIN', 'PTS', 'REB', 'AST']):
            cols[i].metric(stat, f"{prediction[stat]:.1f}", f"{lower[stat]:.1f} – {upper[stat]:.1f}")

    with tabs[1]:
        plot_stat_trend(df, prediction)

    with tabs[2]:
        if opp_name in full_team_stats['TEAM_NAME'].values:
            plot_matchup_difficulty(full_team_stats, opp_name)
        else:
            st.warning("No matchup data available for opponent.")
        plot_archetype_stats(df)
