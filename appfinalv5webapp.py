import streamlit as st
from nba_api.stats.static import players
from nba_api.stats.endpoints import PlayerGameLog
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("🏀 NBA Player Stat Predictor (Upgraded)")

@st.cache_data
def get_player_options():
    return players.get_active_players()

def get_player_id(name):
    result = [p for p in get_player_options() if name.lower() in p["full_name"].lower()]
    if result:
        return result[0]['id'], result[0]['full_name']
    return None, None

@st.cache_data
def load_game_log(player_id):
    return PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]

def preprocess_games(df):
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')
    df['HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['OPPONENT'] = df['MATCHUP'].str.extract(r'@ (\w+)|vs\. (\w+)').bfill(axis=1).iloc[:, 0]
    features = ['MIN', 'PTS', 'REB', 'AST', 'FG3M', 'FGA', 'FG_PCT', 'FG3A',
                'FG3_PCT', 'FTA', 'FT_PCT', 'STL', 'BLK', 'TOV', 'HOME']
    for col in features:
        df[f'{col}_prev'] = df[col].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df

def weighted_train(df, target, quantile):
    features = [col for col in df.columns if '_prev' in col]
    df['weight'] = np.linspace(0.3, 1, len(df))  # heavier for recent
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
    fig.update_layout(height=800, title="📈 Last 15 Games + Predicted")
    st.plotly_chart(fig, use_container_width=True)

def stat_vs_team_bar(df):
    df = df.copy()
    df = df.groupby('OPPONENT')[['PTS', 'REB', 'AST']].mean().sort_values('PTS', ascending=False)
    st.subheader("📊 Avg Stats vs Each Team")
    fig = px.bar(df, x=df.index, y=['PTS', 'REB', 'AST'], barmode='group')
    st.plotly_chart(fig, use_container_width=True)

def timeline_heatmap(df):
    df = df.copy()
    df['Month'] = df['GAME_DATE'].dt.strftime('%b')
    pivot = df.pivot_table(index='OPPONENT', columns='Month', values='PTS', aggfunc='mean')
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap='coolwarm', annot=True, fmt=".1f", linewidths=0.5)
    plt.title("🗓️ PTS Heatmap vs Team by Month")
    st.pyplot(plt.gcf())

# === STREAMLIT UI ===
player_list = [p["full_name"] for p in get_player_options()]
player_name = st.selectbox("Search for a player", sorted(player_list), index=player_list.index("Jayson Tatum"))

if player_name:
    with st.spinner("Loading and predicting..."):
        player_id, full_name = get_player_id(player_name)
        df = load_game_log(player_id)
        df = preprocess_games(df)
        prediction, lower, upper = predict_with_intervals(df)

        st.subheader("🔮 Predicted Stats (with intervals)")
        cols = st.columns(4)
        for i, stat in enumerate(['MIN', 'PTS', 'REB', 'AST']):
            val = prediction[stat]
            low, high = lower[stat], upper[stat]
            cols[i].metric(stat, f"{val:.1f}", f"{low:.1f} – {high:.1f}")

        plot_stat_trend(df, prediction)
        stat_vs_team_bar(df)
        timeline_heatmap(df)
