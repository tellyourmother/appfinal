import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats, LeagueGameFinder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(layout="wide")

@st.cache_data(show_spinner=False)
def get_player_id(player_name):
    found = players.find_players_by_full_name(player_name)
    if not found:
        return None
    return found[0]['id']

@st.cache_data(show_spinner=False)
def get_team_strength():
    df = LeagueDashTeamStats(season='2024-25', season_type_all_star='Regular Season').get_data_frames()[0]
    return df[['TEAM_NAME', 'W_PCT', 'PTS', 'PLUS_MINUS']].rename(columns={
        'W_PCT': 'OPP_W_PCT',
        'PTS': 'OPP_PTS',
        'PLUS_MINUS': 'OPP_PLUS_MINUS'
    })

def get_team_abbr_map():
    return {t['abbreviation']: t['full_name'] for t in teams.get_teams()}

def preprocess(df, team_strength_df):
    abbr_map = get_team_abbr_map()
    df = df.iloc[::-1].reset_index(drop=True)
    df['HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['OPPONENT'] = df['MATCHUP'].apply(lambda x: abbr_map.get(x.split()[-1], x.split()[-1]))
    df = df.merge(team_strength_df, left_on='OPPONENT', right_on='TEAM_NAME', how='left')

    features = ['HOME', 'MIN', 'PTS', 'REB', 'AST', 'FG3M', 'FGA', 'FG_PCT',
                'FG3A', 'FG3_PCT', 'FTA', 'FT_PCT', 'STL', 'BLK', 'TOV',
                'OPP_W_PCT', 'OPP_PTS', 'OPP_PLUS_MINUS']

    df = df[features]
    for col in features:
        df[f'{col}_prev'] = df[col].shift(1)
    df = df.dropna().reset_index(drop=True)

    X = df[[c for c in df.columns if '_prev' in c]]
    y = df[['MIN', 'PTS', 'REB', 'AST']]
    return X, y, df

def train_with_confidence(X, y):
    preds, intervals, models = {}, {}, {}
    for stat in y.columns:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        stack = StackingRegressor(estimators=[('rf', rf), ('xgb', xgb)],
                                  final_estimator=RandomForestRegressor(n_estimators=50, random_state=42))
        stack.fit(X, y[stat])
        pred = stack.predict(X.tail(1))[0]

        y_pred_all = stack.predict(X)
        residuals = y[stat] - y_pred_all
        err = residuals.std()
        preds[stat] = pred
        intervals[stat] = err
        models[stat] = stack
    return preds, intervals, models

def get_next_opponent(player_name, player_id, game_df):
    first_matchup = game_df['MATCHUP'].iloc[0]
    team_abbr = first_matchup.split(' ')[0]
    team_info = next((team for team in teams.get_teams() if team['abbreviation'] == team_abbr), None)
    if not team_info:
        return None
    team_id = team_info['id']
    game_finder = LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable='2024-25',
        season_type_nullable='Regular Season'
    )
    games = game_finder.get_data_frames()[0]
    games = games.sort_values('GAME_DATE').reset_index(drop=True)
    today = pd.Timestamp.now().normalize()
    next_game = games[pd.to_datetime(games['GAME_DATE']) > today].head(1)
    if next_game.empty:
        return None
    matchup = next_game['MATCHUP'].iloc[0]
    opp_abbr = matchup.split()[-1]
    return get_team_abbr_map().get(opp_abbr, opp_abbr)

def plot_player_trend(player_name, game_df, prediction):
    stats_to_plot = ['MIN', 'PTS', 'REB', 'AST']
    game_df = game_df.iloc[::-1].reset_index(drop=True)
    game_df['GAME_DATE'] = pd.to_datetime(game_df['GAME_DATE']).dt.strftime('%b %d')
    game_df['LABEL'] = game_df['GAME_DATE'] + " (" + game_df['MATCHUP'] + ")"

    fig = make_subplots(rows=2, cols=2, subplot_titles=stats_to_plot)
    for i, stat in enumerate(stats_to_plot):
        row = i // 2 + 1
        col = i % 2 + 1
        y_vals = list(game_df[stat]) + [prediction[stat]]
        x_vals = list(game_df['LABEL']) + ['Predicted']
        colors = ['tomato' if val > game_df[stat].mean() else 'skyblue' for val in game_df[stat]] + ['gold']
        rolling = pd.Series(y_vals[:-1]).rolling(window=5).mean()

        fig.add_trace(go.Bar(
            x=x_vals,
            y=y_vals,
            marker_color=colors,
            name=stat,
            showlegend=False,
            hovertemplate="%{x}<br>" + stat + ": %{y}"
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=x_vals[:-1],
            y=rolling,
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Rolling Avg',
            showlegend=(i == 0)
        ), row=row, col=col)

        fig.update_yaxes(title_text=stat, row=row, col=col)

    fig.update_layout(
        title=f"{player_name} - Last 15 Games + Prediction",
        height=800,
        hovermode="x unified",
        plot_bgcolor="white"
    )
    return fig

def create_team_heatmap(player_id):
    all_logs = PlayerGameLog(player_id=player_id, season='ALL').get_data_frames()[0]
    abbr_map = get_team_abbr_map()
    all_logs['TEAM'] = all_logs['MATCHUP'].apply(lambda x: abbr_map.get(x.split()[-1], x.split()[-1]))
    agg = all_logs.groupby('TEAM')[['PTS', 'REB', 'AST', 'MIN']].mean().round(1)

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title("📊 Average Stats vs Each Team")
    heatmap = ax.imshow(agg.values, cmap="coolwarm", aspect='auto')

    ax.set_yticks(np.arange(len(agg.index)))
    ax.set_yticklabels(agg.index)
    ax.set_xticks(np.arange(len(agg.columns)))
    ax.set_xticklabels(agg.columns)

    for i in range(len(agg.index)):
        for j in range(len(agg.columns)):
            ax.text(j, i, agg.values[i, j], ha='center', va='center', color='black')

    plt.colorbar(heatmap)
    plt.tight_layout()
    return fig

# === STREAMLIT UI ===

st.title("🏀 NBA Player Game Predictor")
player_name = st.text_input("Enter full player name", value="Jayson Tatum")

if player_name:
    with st.spinner("Loading data and predicting..."):
        player_id = get_player_id(player_name)
        if not player_id:
            st.error("Player not found.")
        else:
            game_df = PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]
            team_strength_df = get_team_strength()
            X, y, processed_df = preprocess(game_df, team_strength_df)
            prediction, conf, model = train_with_confidence(X, y)
            next_opp = get_next_opponent(player_name, player_id, game_df)

            # Display predicted stats
            st.subheader("🔮 Predicted Next Game Stats")
            if next_opp:
                st.markdown(f"**Next Opponent:** {next_opp}")
            stat_cols = st.columns(4)
            for i, stat in enumerate(['MIN', 'PTS', 'REB', 'AST']):
                val = prediction[stat]
                ci = conf[stat]
                stat_cols[i].metric(stat, f"{val:.1f}", f"± {ci:.1f}")

            # Interactive plot
            st.subheader("📈 Stat Trends (Last 15 Games + Prediction)")
            fig = plot_player_trend(player_name, game_df, prediction)
            st.plotly_chart(fig, use_container_width=True)

            # Heatmap
            st.subheader("🗺️ Heatmap: Avg Stats vs All Teams")
            fig2 = create_team_heatmap(player_id)
            st.pyplot(fig2)
