"""NBA Live JSON and explicitly sourced complete record-snapshot adapters.

Original provider fields remain in data; no undocumented defender assignments,
roster start dates, publication timestamps, or tracking frames are fabricated.
"""

import re
from .timeutil import period_seconds, seconds, utc

BOX = "nba.live.boxscore"
PBP = "nba.live.playbyplay"
GENERIC = "records.v1"


def record(entity, key, data, *, status="observed", event_time=None, updated=None):
    return {"entity": entity, "key": str(key), "status": status,
            "event_time": event_time, "source_updated_at": updated, "data": data}


def normalize(feed: str, scope: str, payload: dict) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")
    if feed == GENERIC:
        if payload.get("schema_version") != 1 or not isinstance(payload.get("records"), list):
            raise ValueError("Expected schema_version=1 and records list")
        return payload["records"]
    if feed not in (BOX, PBP):
        raise ValueError(f"Unsupported feed: {feed}")
    game = payload["game"]
    if not isinstance(game, dict) or str(game["gameId"]) != scope:
        raise ValueError("Payload gameId does not match requested scope")
    if not re.fullmatch(r"\d{10}", scope):
        raise ValueError("NBA game ID must contain ten digits")
    rows = []
    if feed == BOX:
        tipoff = utc(game["gameTimeUTC"])
        # Do not interpret meta.time: its timezone/publication semantics are unclear.
        rows.append(record("games", scope, {key: value for key, value in game.items()
                    if key not in ("homeTeam", "awayTeam")}, event_time=tipoff))
        team_ids = set()
        for side in ("homeTeam", "awayTeam"):
            team = game[side]
            team_id = str(team["teamId"])
            if team_id in team_ids:
                raise ValueError("A game must contain distinct teams")
            team_ids.add(team_id)
            rows.append(record("team_box_scores", team_id,
                        {**{key: value for key, value in team.items() if key != "players"},
                         "side": side, "gameId": scope}, event_time=tipoff))
            for player in team["players"]:
                player_id = str(player["personId"])
                if not isinstance(player.get("statistics"), dict):
                    raise ValueError("Player statistics missing")
                rows.append(record("player_box_scores", player_id,
                            {**player, "teamId": team["teamId"], "gameId": scope}, event_time=tipoff))
                rows.append(record("players", player_id, {key: player[key] for key in
                            ("personId", "name", "firstName", "familyName") if key in player}))
        return rows
    if not isinstance(game["actions"], list):
        raise ValueError("actions must be a list")
    for index, action in enumerate(game["actions"]):
        period = action["period"]
        clock = seconds(action["clock"])
        if clock > period_seconds(period):
            raise ValueError("Clock exceeds period length")
        if not isinstance(action["actionType"], str):
            raise ValueError("actionType must be a string")
        data = {**action, "gameId": scope, "sourceIndex": index}
        rows.append(record("events", action["actionNumber"], data,
                    event_time=action.get("timeActual"), updated=action.get("edited")))
        if action["actionType"] in ("2pt", "3pt", "freethrow"):
            rows.append(record("shots", action["actionNumber"], data,
                        event_time=action.get("timeActual"), updated=action.get("edited")))
    return rows
