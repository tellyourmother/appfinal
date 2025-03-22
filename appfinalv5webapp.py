import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import time
from nba_api.stats.endpoints import LeagueDashTeamStat

st.set_page_config(layout="wide")
st.title("🏀 NBA Player Performance Dashboard")

# ========== CACHED DATA LOADING ==========
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

# ========== MODELING ==========
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

# ========== RADAR CHART ==========
def plot_radar_chart(player_df, league_df, player_name, compare_player=None):
    stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    league_avg = league_df[stats].mean()
    player_avg = player_df[stats].mean()

    scaler = StandardScaler()
    all_data = pd.DataFrame([player_avg, league_avg], index=[player_name, 'League Avg'])
    z_scores = pd.DataFrame(scaler.fit_transform(all_data), columns=stats, index=all_data.index)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=z_scores.loc[player_name].values,
        theta=stats,
        fill='toself',
        name=player_name,
        line=dict(color='gold')
    ))
    fig.add_trace(go.Scatterpolar(
        r=z_scores.loc['League Avg'].values,
        theta=stats,
        fill='toself',
        name='League Avg',
        line=dict(color='gray', dash='dot')
    ))

    if compare_player:
        comp_id, _ = get_player_id(compare_player)
        comp_df = load_game_log(comp_id)
        comp_df = preprocess_games(comp_df, get_team_strength(), get_team_abbr_map())
        comp_avg = comp_df[stats].mean()
        all_data.loc[compare_player] = comp_avg
        comp_z = scaler.transform([comp_avg])[0]

        fig.add_trace(go.Scatterpolar(
            r=comp_z,
            theta=stats,
            fill='toself',
            name=compare_player,
            line=dict(color='deepskyblue')
        ))

    fig.update_layout(
        title="🕸️ Stat Profile (Z-Score Normalized)",
        polar=dict(radialaxis=dict(visible=True, range=[-2.5, 2.5])),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

# ========== VISUALS ==========
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

    st.subheader("🏆 Performance vs Team Tiers")
    fig = px.bar(tier_avg, x='TIER', y=['PTS', 'REB', 'AST'], barmode='group',
                 color_discrete_map={"PTS": "gold", "REB": "skyblue", "AST": "lightgreen"})
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
    prediction, lower, upper = predict_with_intervals(df)

    team_abbr_list = sorted(team_map.keys())
    next_opponent = st.selectbox("Select next opponent team", team_abbr_list)
    st.markdown(f"**Next Opponent:** `{next_opponent}`")

    tabs = st.tabs([
        "📊 Prediction", "📈 Trends", "🆚 Stats vs Teams", 
        "🗺️ Heatmaps", "🏆 Tiers", "📊 Radar Chart"
    ])

    with tabs[0]:
        st.subheader("🔮 Predicted Stats (with confidence)")
        cols = st.columns(4)
        for i, stat in enumerate(['MIN', 'PTS', 'REB', 'AST']):
            cols[i].metric(stat, f"{prediction[stat]:.1f}", f"{lower[stat]:.1f} – {upper[stat]:.1f}")

    with tabs[1]:
        plot_stat_trend(df, prediction)

    with tabs[2]:
        stat_vs_team_bar(df)

    with tabs[3]:
        timeline_heatmap(df)

    with tabs[4]:
        tier_breakdown_chart(df)

    with tabs[5]:
        st.subheader("📊 Stat Profile Radar Chart")
        compare = st.selectbox("Compare with another player (optional)", ["None"] + player_list)
        compare_name = compare if compare != "None" else None

        all_players = get_player_options()
        league_data = []
        for p in all_players[:100]:  # Limit to 100 for speed
            try:
                pid = p['id']
                gdf = load_game_log(pid)
                gdf = preprocess_games(gdf, team_strength, team_map)
                league_data.append(gdf[["PTS", "REB", "AST", "STL", "BLK", "TOV"]].mean())
            except:
                continue
        league_df = pd.DataFrame(league_data)

        plot_radar_chart(df, league_df, player_name, compare_name)


# === Add inside or near get_team_strength ===
@st.cache_data
def get_opponent_difficulty_metrics():
    df = LeagueDashTeamStats(season='2024-25', season_type_all_star='Regular Season').get_data_frames()[0]
    return df[['TEAM_NAME', 'W_PCT', 'DEF_RATING', 'PACE']]

# === Add this helper to calculate difficulty category ===
def categorize_difficulty(def_rating, pace, w_pct):
    if def_rating < 110 and pace < 98 and w_pct > 0.6:
        return "🟥 Hard"
    elif def_rating < 113 or pace < 101 or w_pct > 0.5:
        return "🟨 Medium"
    else:
        return "🟩 Easy"

# === Inside your Streamlit UI section (after load_game_log etc)
opp_metrics = get_opponent_difficulty_metrics()
team_abbr_map = get_team_abbr_map()
opp_name = team_abbr_map.get(next_opponent, next_opponent)
opp_info = opp_metrics[opp_metrics['TEAM_NAME'] == opp_name]

st.markdown("### 🎯 Matchup Difficulty")
if not opp_info.empty:
    row = opp_info.iloc[0]
    difficulty = categorize_difficulty(row['DEF_RATING'], row['PACE'], row['W_PCT'])
    st.write(f"**{opp_name}**")
    st.write(f"- **DEF_RATING:** {row['DEF_RATING']:.1f}")
    st.write(f"- **PACE:** {row['PACE']:.1f}")
    st.write(f"- **W_PCT:** {row['W_PCT']:.2f}")
    st.subheader(f"🏀 Matchup Difficulty: {difficulty}")
else:
    st.warning("Opponent difficulty info not found.")

# === NEW TAB ===
tabs = st.tabs([
    "📊 Prediction", "📈 Trends", "🆚 Stats vs Teams", 
    "🗺️ Heatmaps", "🏆 Tiers", "📊 Radar Chart", "🧠 Matchup Intelligence"
])

# === TAB 6: Matchup Intelligence ===
with tabs[6]:
    st.subheader("📊 Performance vs Opponent Archetypes")
    df_intel = df.copy()

    # Pace tier
    df_intel['PACE_TIER'] = pd.qcut(df_intel['PACE'], q=3, labels=['Slow', 'Medium', 'Fast'])
    pace_avg = df_intel.groupby('PACE_TIER')[['PTS', 'REB', 'AST']].mean().reset_index()
    st.markdown("#### 🏃‍♂️ By Opponent Pace")
    fig1 = px.bar(pace_avg, x='PACE_TIER', y=['PTS', 'REB', 'AST'], barmode='group')
    st.plotly_chart(fig1, use_container_width=True)

    # Defense tier
    df_intel['DEF_TIER'] = pd.qcut(df_intel['DEF_RATING'], q=3, labels=['Strong D', 'Average D', 'Weak D'])
    def_avg = df_intel.groupby('DEF_TIER')[['PTS', 'REB', 'AST']].mean().reset_index()
    st.markdown("#### 🧱 By Opponent Defense Rating")
    fig2 = px.bar(def_avg, x='DEF_TIER', y=['PTS', 'REB', 'AST'], barmode='group')
    st.plotly_chart(fig2, use_container_width=True)

def preprocess_games(df, team_strength_df, team_map):
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')
    df['HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['OPP_ABBR'] = df['MATCHUP'].str.extract(r'@ (\w+)|vs\. (\w+)').bfill(axis=1).iloc[:, 0]
    df['OPPONENT'] = df['OPP_ABBR'].map(team_map)

    # Merge opponent difficulty
    full_metrics = LeagueDashTeamStats(season='2024-25', season_type_all_star='Regular Season').get_data_frames()[0]
    df = df.merge(full_metrics[['TEAM_NAME', 'W_PCT', 'DEF_RATING', 'PACE']], 
                  left_on='OPPONENT', right_on='TEAM_NAME', how='left')
    df['W_PCT'] = df['W_PCT'].fillna(0.5)

    # Continue as before
    features = ['MIN', 'PTS', 'REB', 'AST', 'FG3M', 'FGA', 'FG_PCT', 'FG3A',
                'FG3_PCT', 'FTA', 'FT_PCT', 'STL', 'BLK', 'TOV', 'HOME']
    for col in features:
        df[f'{col}_prev'] = df[col].shift(1)
    return df.dropna().reset_index(drop=True)
