"""Optional integration test against a pinned, real archived NBA response pair.

No downloads or fabricated production data. See docs/validation.md for source.
"""

import hashlib
import os
from pathlib import Path
import unittest

from nba_foundation import Archive
from nba_foundation.adapters import BOX, PBP
from nba_foundation.reconcile import reconcile_game
from nba_foundation.timeutil import now

COMMIT = "e7ccf2fb6326da630a3cf1665aac964a1e108f4c"
FILES = (
    (BOX, "game_details/live_0022000001.json", "93bcd1352a4c5a807a7515f9f5745181b0ec5795f6126a274fe8d472e97debc3"),
    (PBP, "pbp/live_0022000001.json", "98d1c6c3f2ba1762880252419bd59d41427a6d5bd37d617d9dff2efba9f26349"),
)


@unittest.skipUnless(os.environ.get("NBA_REFERENCE_DIR"), "Set NBA_REFERENCE_DIR to the pinned pbpstats tests/data directory")
class ReferenceGameTest(unittest.TestCase):
    def test_archived_real_game(self):
        root = Path(os.environ["NBA_REFERENCE_DIR"])
        with Archive(":memory:") as archive:
            for feed, path, expected_hash in FILES:
                raw = (root / path).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_hash, "Reference bytes changed")
                snapshot = archive.ingest(raw, source="nba", feed=feed, scope="0022000001", data_class="real",
                    source_uri=f"https://github.com/dblackrun/pbpstats/blob/{COMMIT}/tests/data/{path}")
                self.assertIsNone(archive.metadata(snapshot)["normalization_error"])
            report = reconcile_game(archive, "0022000001", now())
            self.assertTrue(report["passed"], report["issues"])
            self.assertEqual(len(report["checks"]), 520)
            self.assertEqual(len(report["lineups"]["stints"]), 37)
            self.assertEqual(sum(report["lineups"]["player_seconds"].values()), 28800)
