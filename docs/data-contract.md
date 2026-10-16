# Data contract, version 1

## Snapshot identity and time

A snapshot is identified by `(source, feed, scope, id)`. The monotonically
increasing SQLite `id` breaks ties between receipts with the same timestamp.
`scope` must stay stable for a complete feed response, such as an NBA game ID or
a provider's league-wide injury-report snapshot. Do not import record deltas
under a complete-snapshot scope. The latest eligible receipt replaces that
scope's entire record set, including deletions. Ordering is by receipt, not an
inferred source revision number: a connector must not send old cached responses
as newer corrections without identifying that limitation.

`retrieved_at` is generated when the archive receives bytes. The CLI does not
allow backdating it. `source_published_at` is optional, timezone-aware source
metadata. `source_updated_at` belongs to an individual record. `event_time` is
when a fact occurred or a game is scheduled; it is not proof of publication.

`available_at` is the maximum of receipt, supplied publication, and all supplied
record-update timestamps. The entire snapshot is withheld until that time.
This is conservative when a source supplies future or inconsistent edit times.
All timestamps are normalized to UTC with microsecond precision. A timestamp
without an offset is rejected or quarantined, never interpreted as local time.

This implementation reconstructs **what this archive had received**, not all
information that might have existed publicly. Historical files collected today
cannot establish a historical information set. Verified legacy capture logs and
a separately reviewed migration process would be needed to import earlier
receipt histories. Record corrections are never backdated to event time.

## Storage

`snapshots` stores raw bytes, SHA-256, classification, source URI, selected
response headers, timestamps, parser version, and normalization errors.
`records` stores `(snapshot_id, entity, record_key, status, event_time,
source_updated_at, data_json)`. Duplicate keys inside an entity/snapshot reject
the complete normalization transaction. Invalid JSON or unsupported schemas
retain raw bytes with a quarantine reason and zero normalized records.

Named SQL views expose `games`, `players`, `roster_history`,
`availability_reports`, `events`, `shots`, `team_box_scores`, `player_box_scores`,
`id_mappings`, `lineup_stints`, `possessions`, `action_tags`, `tracking_frames`,
and `prediction_snapshots`. These views show **all versions** for auditing;
they are not safe feature queries. Use `Archive.records_as_of` or the CLI for
cutoff-safe access. JSON retains provider fields without collapsing percentages
or dropping attempt counts. No cross-provider ID equivalence is guessed.

The `real`/`test`/`scrambled`/`unverified` classification applies to the source
payload. The separate record status describes the epistemic status of a value:

| Status | Meaning |
| --- | --- |
| observed | Recorded by the named source |
| derived | Computed from identified source observations |
| estimated | A statistical estimate, with method/uncertainty supplied in data |
| simulated | Generated as part of a possible future |
| unavailable | Explicitly missing; never interpreted as zero |

Status is preserved on export. Consumers must select appropriate statuses for
their task; an estimated or simulated record is not transformed into an
observation merely because its source document is classified `real`.

## Generic complete-snapshot ingestion

Use `--feed records.v1`, a stable provider `--source` and `--scope`, and the real
source URI. The JSON envelope has `schema_version: 1` and `records: []`. Each
record requires:

| Field | Type / requirement |
| --- | --- |
| entity | One of the named entities above |
| key | Nonempty string, unique within entity and snapshot |
| status | One of the five statuses above |
| data | JSON object containing the original IDs and values |
| event_time | Optional timestamp with timezone, or null |
| source_updated_at | Optional documented timestamp with timezone, or null |

The adapter validates the envelope, timestamps, finite JSON values, identities,
and statuses. It does **not** certify vendor-specific field semantics. Provider
adapters must define and validate those before feature construction. For roster
history retain effective membership bounds separately from publication time;
for availability retain player/game IDs, designation, report timing, and source;
for ID mappings retain both provider IDs, canonical ID, effective dates, and
evidence. Unknown fields stay null/absent, not inferred values.

`prediction_snapshots` is a reserved archival entity; no prediction service is
implemented. Its eventual typed contract must require team IDs, scheduled
tipoff, venue/home/away/neutral status, competition/rule version, information
cutoff, prediction mode, input snapshot IDs, model version, and output statuses.

## NBA adapters

The supported network API is the **NBA Live JSON family**, not the deprecated
stats `PlayByPlayV2` endpoint. `nba.live.boxscore` maps game details, players, and
team/player box scores. `nba.live.playbyplay` maps actions and shot records,
retaining `actionNumber`, `orderNumber`, period, clock, source IDs, `timeActual`,
`edited`, and any supplied coordinates. Missing coordinates stay missing.

Relevant primary references:

- [NBA Live box-score adapter](https://github.com/swar/nba_api/blob/master/src/nba_api/live/nba/endpoints/boxscore.py)
- [NBA Live play-by-play adapter](https://github.com/swar/nba_api/blob/master/src/nba_api/live/nba/endpoints/playbyplay.py)
- [NBA Live HTTP base URL](https://github.com/swar/nba_api/blob/master/src/nba_api/live/nba/library/http.py)
- [NBA rules](https://official.nba.com/rulebook/)

Reconstructed lineups are derived output in the reconciliation report, with both
input snapshot IDs. First-period starters come from box-score starting flags;
later-period starters require five provable participants before their first
recorded sub-in. Same-clock substitution batches are applied together. Missing
evidence, invalid counts, or an unsupported sequence blocks the gate. These are
postgame reconstructions; using them as pregame observations would be leakage.

SQLite triggers prevent accidental mutation through normal SQL, not tampering
by someone who can replace the database or drop triggers. Back up the database
with SQLite's backup API; raw archives can contain licensed data and must not
be committed to this repository. Scale-out storage and migrations beyond
schema version 1 are future work.
