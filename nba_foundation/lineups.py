"""Conservative postgame reconstruction; ambiguous periods remain unresolved."""

from collections import defaultdict
from itertools import groupby

from .timeutil import period_seconds, seconds


def ordered_events(events):
    return sorted(events, key=lambda event: (event["period"],
                  event.get("orderNumber", event["sourceIndex"]), event["sourceIndex"]))


def reconstruct(events: list[dict], players: list[dict], last_period: int) -> dict:
    teams = defaultdict(set)
    owner = {}
    for player in players:
        pid, tid = str(player["personId"]), str(player["teamId"])
        teams[tid].add(pid)
        owner[pid] = tid
    issues, stints = [], []
    totals = defaultdict(float)
    events = ordered_events(events)
    for period in range(1, last_period + 1):
        actions = [event for event in events if event["period"] == period]
        length = period_seconds(period)
        if not any(event["actionType"] == "period" and event.get("subType") == "start"
                   and seconds(event["clock"]) == length for event in actions):
            issues.append(f"period {period}: missing period-start event")
            continue
        if not actions or not any(event["actionType"] == "period" and event.get("subType") == "end"
                                  and seconds(event["clock"]) == 0 for event in actions):
            issues.append(f"period {period}: missing period-end event")
            continue
        for team_id, roster in teams.items():
            if period == 1:
                active = {str(p["personId"]) for p in players
                          if str(p["teamId"]) == team_id and str(p.get("starter")) == "1"}
            else:
                # A first on-court involvement or sub-out, before any sub-in,
                # proves the player started this period. Never fill missing slots.
                seen, active = set(), set()
                for action in actions:
                    pid = str(action.get("personId", 0))
                    if action["actionType"] == "substitution":
                        if pid in roster and pid not in seen:
                            seen.add(pid)
                            if action.get("subType") == "out":
                                active.add(pid)
                        continue
                    participants = []
                    if action["actionType"] in ("2pt", "3pt", "freethrow", "rebound", "turnover"):
                        participants.append(pid)
                    participants += [str(action[key]) for key in
                                     ("assistPersonId", "blockPersonId", "stealPersonId") if action.get(key)]
                    for participant in participants:
                        if participant in roster and participant not in seen:
                            seen.add(participant)
                            active.add(participant)
            if len(active) != 5:
                issues.append(f"period {period}, team {team_id}: {len(active)} provable starters; expected 5")
                continue
            period_stints, remaining, failed = [], float(length), False
            for clock, group in groupby(actions, key=lambda event: seconds(event["clock"])):
                group = list(group)
                if clock > remaining:
                    issues.append(f"period {period}: non-monotonic game clock")
                    failed = True
                    break
                if remaining > clock:
                    period_stints.append({"team_id": team_id, "period": period,
                                          "start_clock_seconds": remaining, "end_clock_seconds": clock,
                                          "player_ids": sorted(active), "status": "derived"})
                substitutions = [a for a in group if a["actionType"] == "substitution"
                                 and str(a.get("teamId")) == team_id]
                outs = [str(a.get("personId")) for a in substitutions if a.get("subType") == "out"]
                ins = [str(a.get("personId")) for a in substitutions if a.get("subType") == "in"]
                if (len(outs) != len(ins) or len(set(outs)) != len(outs) or len(set(ins)) != len(ins)
                        or set(outs) & set(ins) or not set(outs) <= active
                        or set(ins) & active or not set(ins) <= roster
                        or len(substitutions) != len(outs) + len(ins)):
                    issues.append(f"period {period}, team {team_id}, clock {clock}: unresolved substitution batch")
                    failed = True
                    break
                after = (active - set(outs)) | set(ins)
                # Same-clock event ordering can place an action before or after
                # a substitution. Either membership is acceptable for a zero-time event.
                for action in group:
                    if action["actionType"] in ("2pt", "3pt", "freethrow", "rebound", "turnover"):
                        pid = str(action.get("personId", 0))
                        if owner.get(pid) == team_id and pid not in active | after:
                            issues.append(f"period {period}, team {team_id}: event actor {pid} is off court")
                            failed = True
                active, remaining = after, clock
            if failed:
                continue
            # A stint is a continuous lineup, not one row for every event clock.
            merged = []
            for stint in period_stints:
                if (merged and merged[-1]["player_ids"] == stint["player_ids"]
                        and merged[-1]["end_clock_seconds"] == stint["start_clock_seconds"]):
                    merged[-1]["end_clock_seconds"] = stint["end_clock_seconds"]
                else:
                    merged.append(dict(stint))
            stints.extend(merged)
            for stint in period_stints:
                duration = stint["start_clock_seconds"] - stint["end_clock_seconds"]
                for pid in stint["player_ids"]:
                    totals[pid] += duration
    return {"status": "unresolved" if issues else "reconstructed", "issues": issues,
            "stints": stints, "player_seconds": dict(totals)}
