"""SYNTHETIC unit-test cases only. Never use for training or accuracy evaluation."""

from collections import Counter
from nba_foundation.reconcile import COUNTS

GAME_ID = "0020000001"
T0 = "2025-01-01T00:00:00Z"
T1 = "2025-01-02T00:00:00Z"
T2 = "2025-01-03T00:00:00Z"


def full_game(overtime=False):
    periods = 5 if overtime else 4
    score = [0, 0]
    individual = {pid: Counter({key: 0 for key in COUNTS}) for pid in range(1, 11)}
    events = []

    def add(period, clock, kind, **fields):
        number = len(events) + 1
        events.append({"actionNumber": number, "orderNumber": number * 10000,
                       "period": period, "clock": clock, "actionType": kind,
                       "teamId": 0, "personId": 0, "scoreHome": str(score[0]),
                       "scoreAway": str(score[1]), **fields})

    for period in range(1, periods + 1):
        add(period, "PT12M00S" if period <= 4 else "PT05M00S", "period", subType="start")
        # Every later period has enough participation evidence to prove its starters.
        for pid in ([1] if period == 1 else range(1, 11)):
            side = 0 if pid <= 5 else 1
            score[side] += 2
            individual[pid].update({"points": 2, "fieldGoalsMade": 1, "fieldGoalsAttempted": 1})
            add(period, f"PT02M{50-pid:02d}S", "2pt", personId=pid, teamId=100 + side,
                shotResult="Made")
        add(period, "PT00M00S", "period", subType="end")
    teams = []
    for side in (0, 1):
        members = [{"personId": pid, "name": f"TEST PLAYER {pid}", "starter": "1",
                    "statistics": {**individual[pid], "minutes": f"PT{48 + (5 if overtime else 0)}M00S"}}
                   for pid in range(1 + side * 5, 6 + side * 5)]
        sums = {stat: sum(p["statistics"][stat] for p in members) for stat in COUNTS}
        teams.append({"teamId": 100 + side, "score": score[side], "players": members,
                      "statistics": {**sums, "reboundsPersonal": 0, "reboundsTeam": 0,
                                     "reboundsTotal": 0, "turnoversTeam": 0, "turnoversTotal": 0}})
    return ({"game": {"gameId": GAME_ID, "gameTimeUTC": T0, "gameStatus": 3,
                      "regulationPeriods": 4, "period": periods, "homeTeam": teams[0], "awayTeam": teams[1]}},
            {"game": {"gameId": GAME_ID, "actions": events}})


def reports(status="questionable"):
    return {"schema_version": 1, "records": [
        {"entity": "availability_reports", "key": "test-player", "status": "observed",
         "event_time": T0, "data": {"player_id": "test-player", "designation": status}}
    ]}
