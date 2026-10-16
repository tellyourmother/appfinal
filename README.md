# NBA player dashboard and auditable data foundation

`appfinalv5webapp.py` is the existing Streamlit player dashboard. `nba_foundation/`
adds the first backend increment toward a point-in-time basketball forecasting
system: immutable source snapshots, conservative cutoff queries, and final-game
reconciliation. The new backend uses only the Python standard library (3.11+).

## Run the data foundation

From the repository root:

```sh
python -m nba_foundation --db nba_archive.sqlite3 init
python -m nba_foundation --db nba_archive.sqlite3 fetch-game 0022000001
```

The fetch command requests the official NBA Live JSON box score and play-by-play.
It stores the exact bytes, SHA-256 hash, source URI, selected HTTP headers,
ingestion timestamp, parser version, and normalized records. Source failures
produce a nonzero exit code and a `withheld` message. There is no sample-data
fallback. If the first response succeeds and the second fails, the first remains
archived; reconciliation requires both inputs.

To import a response you already obtained from a real source:

```sh
python -m nba_foundation --db nba_archive.sqlite3 ingest /path/to/boxscore.json \
  --source nba --feed nba.live.boxscore --scope 0022000001 \
  --source-uri 'https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022000001.json' \
  --data-class real
python -m nba_foundation --db nba_archive.sqlite3 ingest /path/to/playbyplay.json \
  --source nba --feed nba.live.playbyplay --scope 0022000001 \
  --source-uri 'https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_0022000001.json' \
  --data-class real
```

Local files default to `unverified`. `--data-class real` is the operator's
attestation about the file's origin, not an automated authenticity check. Never
use it for synthetic or scrambled vendor trial data. Sources remain subject to
their access terms and actual historical coverage.

All reads require a timezone-aware cutoff. For example, **replace the timestamp
below with the actual forecast or audit cutoff**:

```sh
python -m nba_foundation --db nba_archive.sqlite3 query \
  --source nba --feed nba.live.playbyplay --scope 0022000001 \
  --cutoff '2026-10-16T12:00:00Z' --entity events
python -m nba_foundation --db nba_archive.sqlite3 reconcile 0022000001 \
  --cutoff '2026-10-16T12:00:00Z'
```

These commands cannot see a response collected after that cutoff, even if the
game happened years earlier. The historical example ID identifies a game; it
does not imply that fetching it today creates a usable pregame history.

`reconcile` emits JSON with input snapshot IDs, every supported comparison,
unresolved issues, derived lineup stints, and player seconds. It exits `0` only
when all supported checks pass; failures or unavailable data exit `2`. Passing
this limited gate does not validate all NBA rules or prediction accuracy.

## Guarantees and current limits

- Exact raw responses are immutable through the archive interface and SQLite
  update/delete triggers. Malformed responses are retained and quarantined.
- Observed, derived, estimated, simulated, and unavailable are distinct record
  statuses. Query results retain those labels and all source identifiers.
- Each source/feed/scope is a **complete replacement snapshot**. Corrections
  append versions; records removed by a correction cannot reappear in queries.
- Visibility requires ingestion, documented source publication, and supplied
  record-update times to be at or before the cutoff. Unknown publication time
  remains null. NBA `meta.time` is not assigned an undocumented timezone.
- Test, scrambled, unverified, quarantined, missing, and explicitly stale inputs
  are withheld. `query --max-age-seconds 300` applies a five-minute ingestion-age
  limit. Fresh ingestion does not itself prove that a source is current.
- Box-score team membership and starting flags are observations about that
  game. They are **not** inferred historical roster membership or pregame injury
  information. A final box score must not become a pregame availability report.
- Supported reconciliation covers shooting, scores, assists, steals, blocks,
  personal/team rebounds and turnovers, five-player lineups, and minutes.
  Ambiguous period starters and substitution batches remain unresolved.
- Possession boundaries, foul taxonomy, rebound eligibility, tracking, forecasts,
  calibration, and live-state recovery are not implemented in this increment.

Availability reports, roster history, ID mappings, and other sourced records can
be archived through the documented `records.v1` interface. No vendor-specific
injury connector or historical availability dataset is bundled. See
[the data contract](docs/data-contract.md) and [validation notes](docs/validation.md).

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests use explicitly labeled synthetic cases to check software invariants. These
are not basketball training data or evidence of forecast accuracy. The separate
real-game smoke check in `docs/validation.md` documents its source and limits.

## Existing dashboard

The Streamlit entry point continues to use its original data path. It has not
yet been connected to the new archive, so its existing prediction display is
**not** a validated cutoff-safe forecast. It uses a fixed 2024–25 season, its
model evaluation covers one held-out row, and its displayed model spread is not
a calibrated predictive interval. The selected next opponent is not currently
an input to those models. Stage 2 should replace this workflow with explicit
game date, venue, competition, and cutoff inputs and chronological evaluation.

No claim of model quality or improvement is made by this change.
