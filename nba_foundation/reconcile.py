"""Compare the event ledger, player box scores, team totals, and lineup minutes.

This gate checks the supported metrics explicitly; it is not a claim of complete
NBA rule validation or possession reconstruction.
"""

from collections import Counter, defaultdict
import json

from .adapters import BOX, PBP
from .lineups import ordered_events, reconstruct
from .timeutil import seconds, utc

SHOOTING = ("points", "fieldGoalsAttempted", "fieldGoalsMade", "threePointersAttempted",
            "threePointersMade", "freeThrowsAttempted", "freeThrowsMade")
COUNTS = SHOOTING + ("assists", "blocks", "steals", "reboundsOffensive", "reboundsDefensive", "turnovers")


def reconcile_game(archive, game_id: str, cutoff: str) -> dict:
    box = archive.snapshot_as_of(source="nba", feed=BOX, scope=game_id, cutoff=cutoff)
    pbp = archive.snapshot_as_of(source="nba", feed=PBP, scope=game_id, cutoff=cutoff)
    game = json.loads(archive.raw(box["id"]))["game"]
    # Use the normalized, versioned actions for stable ordering metadata.
    events = [r["data"] for r in archive.records_as_of(source="nba", feed=PBP, scope=game_id,
              cutoff=cutoff, entity="events")["records"]]
    events = ordered_events(events)
    teams = [game["homeTeam"], game["awayTeam"]]
    players = [{**player, "teamId": team["teamId"]} for team in teams for player in team["players"]]
    owner = {str(player["personId"]): str(player["teamId"]) for player in players}
    team_ids = {str(team["teamId"]) for team in teams}
    ledger = defaultdict(Counter)
    team_rebounds, team_turnovers = Counter(), Counter()
    issues, checks = [], []

    def compare(label, actual, expected, tolerance=0):
        valid = isinstance(actual, (int, float)) and isinstance(expected, (int, float))
        passed = valid and abs(actual - expected) <= tolerance
        checks.append({"check": label, "actual": actual, "expected": expected, "passed": passed})
        if not passed:
            issues.append(label)

    if game.get("gameStatus") != 3:
        issues.append("Game is not final; final-game reconciliation withheld")
    period = game.get("period")
    if type(period) is not int or period < 4 or game.get("regulationPeriods") != 4:
        issues.append("Unsupported or incomplete regulation/period metadata")
        period = 4
    if events:
        compare("home final event score", int(events[-1]["scoreHome"]), teams[0]["score"])
        compare("away final event score", int(events[-1]["scoreAway"]), teams[1]["score"])
        compare("last event period", events[-1]["period"], period)
        compare("final clock", seconds(events[-1]["clock"]), 0)
    else:
        issues.append("No play-by-play events")

    for event in events:
        kind, pid, tid = event["actionType"], str(event.get("personId", 0)), str(event.get("teamId", 0))
        if kind in ("2pt", "3pt", "freethrow", "rebound", "turnover"):
            if tid not in team_ids:
                issues.append(f"event {event['actionNumber']}: unknown team {tid}")
            if pid != "0" and owner.get(pid) != tid:
                issues.append(f"event {event['actionNumber']}: player/team mismatch")
        if kind in ("2pt", "3pt", "freethrow"):
            result = event.get("shotResult")
            if result not in ("Made", "Missed") or pid not in owner:
                issues.append(f"event {event['actionNumber']}: unknown shot result or shooter")
                continue
            made = result == "Made"
            if kind == "freethrow":
                ledger[pid]["freeThrowsAttempted"] += 1
                ledger[pid]["freeThrowsMade"] += int(made)
                ledger[pid]["points"] += int(made)
            else:
                ledger[pid]["fieldGoalsAttempted"] += 1
                ledger[pid]["fieldGoalsMade"] += int(made)
                ledger[pid]["points"] += (3 if kind == "3pt" else 2) * int(made)
                if kind == "3pt":
                    ledger[pid]["threePointersAttempted"] += 1
                    ledger[pid]["threePointersMade"] += int(made)
                assist, block = str(event.get("assistPersonId", 0)), str(event.get("blockPersonId", 0))
                if assist != "0":
                    if not made or assist == pid or owner.get(assist) != tid:
                        issues.append(f"event {event['actionNumber']}: invalid assist credit")
                    ledger[assist]["assists"] += 1
                if block != "0":
                    if made or block not in owner or owner[block] == tid:
                        issues.append(f"event {event['actionNumber']}: invalid block credit")
                    ledger[block]["blocks"] += 1
        elif kind == "rebound":
            if pid == "0":
                team_rebounds[tid] += 1
            elif event.get("subType") in ("offensive", "defensive"):
                stat = "reboundsOffensive" if event["subType"] == "offensive" else "reboundsDefensive"
                ledger[pid][stat] += 1
            else:
                issues.append(f"event {event['actionNumber']}: unknown rebound type")
        elif kind == "turnover":
            if pid == "0":
                team_turnovers[tid] += 1
            else:
                ledger[pid]["turnovers"] += 1
            steal = str(event.get("stealPersonId", 0))
            if steal != "0":
                if steal not in owner or owner[steal] == tid:
                    issues.append(f"event {event['actionNumber']}: invalid steal credit")
                ledger[steal]["steals"] += 1

    expected_seconds = 5 * (2880 + 300 * (period - 4))
    for team in teams:
        tid, stats = str(team["teamId"]), team["statistics"]
        members = team["players"]
        for stat in COUNTS:
            reported = [p["statistics"].get(stat) for p in members]
            summed = sum(reported) if all(isinstance(v, (int, float)) for v in reported) else None
            compare(f"team {tid}: player sum {stat}", summed, stats.get(stat))
        compare(f"team {tid}: scoreboard points", stats.get("points"), team["score"])
        if all(isinstance(stats.get(k), (int, float)) for k in
               ("fieldGoalsMade", "threePointersMade", "freeThrowsMade")):
            compare(f"team {tid}: shooting points", 2 * stats["fieldGoalsMade"] + stats["threePointersMade"]
                    + stats["freeThrowsMade"], team["score"])
        compare(f"team {tid}: team rebounds", team_rebounds[tid], stats.get("reboundsTeam"))
        compare(f"team {tid}: team turnovers", team_turnovers[tid], stats.get("turnoversTeam"))
        if all(isinstance(stats.get(k), (int, float)) for k in ("reboundsPersonal", "reboundsTeam")):
            compare(f"team {tid}: total rebounds", stats["reboundsPersonal"] + stats["reboundsTeam"], stats.get("reboundsTotal"))
        if all(isinstance(stats.get(k), (int, float)) for k in ("turnovers", "turnoversTeam")):
            compare(f"team {tid}: total turnovers", stats["turnovers"] + stats["turnoversTeam"], stats.get("turnoversTotal"))
        try:
            # Provider durations may be rounded to whole seconds per player.
            compare(f"team {tid}: player minutes", sum(seconds(p["statistics"]["minutes"]) for p in members),
                    expected_seconds, tolerance=len(members))
        except (ValueError, KeyError, TypeError):
            issues.append(f"team {tid}: missing/invalid player minutes")
        for player in members:
            pid = str(player["personId"])
            for stat in COUNTS:
                compare(f"player {pid}: ledger {stat}", ledger[pid][stat], player["statistics"].get(stat))

    lineups = reconstruct(events, players, period)
    issues.extend(lineups["issues"])
    if lineups["status"] == "reconstructed":
        for player in players:
            pid = str(player["personId"])
            try:
                compare(f"player {pid}: lineup seconds", lineups["player_seconds"].get(pid, 0),
                        seconds(player["statistics"]["minutes"]), tolerance=1)
            except (ValueError, KeyError, TypeError):
                issues.append(f"player {pid}: invalid minutes")
    return {"game_id": game_id, "cutoff": utc(cutoff), "passed": not issues,
            "box_snapshot_id": box["id"], "pbp_snapshot_id": pbp["id"],
            "issues": issues, "checks": checks, "lineups": lineups,
            "coverage": "Shooting, assists, blocks, steals, rebounds, turnovers, score, minutes and lineups. "
                        "Foul taxonomy, rebound eligibility, possession boundaries and full rule validation are pending."}
