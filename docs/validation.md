# Validation of the first data-foundation increment

Validated on 2026-10-16 using Python 3.12.14. This document reports software and
historical reconciliation checks, not forecasting performance.

## Automated checks

`python -m unittest discover -s tests -v` runs 25 offline unit/CLI checks and
discovers one optional integration test. The integration test skips unless the
external reference files are explicitly supplied. Unit cases are labeled
synthetic, live only under `tests/`, and must never enter a training/evaluation
dataset. They exercise:

- Collection after a historical cutoff; original versus corrected snapshots;
  source publication/edit times; timezone normalization; removed records.
- Immutable raw bytes, atomic quarantine, invalid JSON, duplicate records,
  nonfinite normalized values, and explicit data-status labels.
- Test/scrambled/unverified data exclusion and freshness limits.
- Score/stat mismatches, missing/duplicate events, foreign game IDs,
  unresolved starters, paired/unpaired substitutions, final-game status,
  regulation and overtime player-minute totals.
- On-disk persistence, CLI error behavior, and HTTP failures without invented
  fallback data.

## Real archived game

The optional integration test uses NBA game `0022000001`, a completed game with
official scores **125–99**, preserved in the `pbpstats` project's test corpus.
It is a third-party archive of real source responses, not a live fetch or proof
of historic availability at a pregame cutoff.

Pinned upstream revision:
[`e7ccf2fb6326da630a3cf1665aac964a1e108f4c`](https://github.com/dblackrun/pbpstats/tree/e7ccf2fb6326da630a3cf1665aac964a1e108f4c)

| Input | SHA-256 |
| --- | --- |
| `tests/data/game_details/live_0022000001.json` | `93bcd1352a4c5a807a7515f9f5745181b0ec5795f6126a274fe8d472e97debc3` |
| `tests/data/pbp/live_0022000001.json` | `98d1c6c3f2ba1762880252419bd59d41427a6d5bd37d617d9dff2efba9f26349` |

Result: **610 source events, 520 supported comparisons passed, zero unresolved
issues, 37 derived lineup stints, and 28,800 total player-seconds across both
teams** (240 player-minutes per team). The files are not bundled in this repo.
Their upstream availability and usage rights remain separate from this code.

To reproduce, obtain that revision of `dblackrun/pbpstats` and point at its
`tests/data` directory:

```sh
NBA_REFERENCE_DIR=/path/to/pbpstats/tests/data python -m unittest discover -s tests -v
```

With the pinned inputs present, all **26 tests** passed. The integration test
verifies both hashes before ingesting the responses. It uses the actual current
ingestion time and queries afterward; it does not backdate either file.

## External-source limitation

A direct request to NBA's Live box-score endpoint returned **HTTP 403** in the
implementation environment. The downloader follows the documented NBA Live
endpoint shape, but its successful operation against the live source was not
verified here. The archive can ingest authorized saved responses. No access
restriction was bypassed and no scrambled/synthetic replacement was used for
the real-game check.

## Gate before Stage 2

This first increment is not the entire Stage 1 research-data program. Before
training or publishing a forecast:

1. Establish permitted historical/current source access and validate multiple
   seasons, including corrections, unusual games, and unresolved-lineup rates.
2. Add provider-specific roster/availability adapters and accumulate actual
   timestamped snapshots. The generic archive is not a historical injury feed.
3. Add full possession reconstruction, foul/free-throw taxonomy, rebound
   eligibility checks, and explicit handling of rare rule exceptions.
4. Specify game/venue/competition/cutoff inputs and replace the dashboard's
   legacy direct-read prediction path with cutoff-safe features.
5. Validate feature definitions and data quality before chronological model
   training, calibration, and comparisons against simple baselines.
