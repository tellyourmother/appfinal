import copy
import json
import sqlite3
import unittest

from nba_foundation.adapters import BOX, GENERIC, PBP
from nba_foundation.reconcile import reconcile_game
from nba_foundation.store import Archive, DataUnavailable
from nba_foundation.timeutil import utc
from tests.fixtures import GAME_ID, T0, T1, T2, full_game, reports


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.time = T1
        self.archive = Archive(":memory:", clock=lambda: self.time)
        self.query = dict(source="TEST-SOURCE", feed=GENERIC, scope="daily", cutoff=T1)

    def tearDown(self):
        self.archive.__exit__()

    def ingest(self, payload, **kwargs):
        # 'real' is mocked only to exercise production gates. All fixture contents
        # are synthetic and the source name explicitly identifies them as tests.
        return self.archive.ingest(json.dumps(payload).encode(), source="TEST-SOURCE",
                                   feed=GENERIC, scope="daily", data_class=kwargs.pop("data_class", "real"), **kwargs)

    def test_ingestion_time_prevents_historical_backfill_leakage(self):
        self.ingest(reports(), source_published_at=T0)
        with self.assertRaises(DataUnavailable):
            self.archive.records_as_of(**{**self.query, "cutoff": T0})
        self.assertEqual(len(self.archive.records_as_of(**self.query)["records"]), 1)

    def test_corrections_do_not_rewrite_earlier_forecasts(self):
        original = self.ingest(reports())
        self.time = T2
        correction = self.ingest(reports("out"), source_published_at=T0)
        before = self.archive.records_as_of(**self.query)
        after = self.archive.records_as_of(**{**self.query, "cutoff": T2})
        self.assertEqual(before["snapshot"]["id"], original)
        self.assertEqual(after["snapshot"]["id"], correction)
        self.assertEqual(before["records"][0]["data"]["designation"], "questionable")
        self.assertEqual(after["records"][0]["data"]["designation"], "out")

    def test_deleted_records_do_not_resurrect(self):
        self.ingest(reports())
        self.time = T2
        self.ingest({"schema_version": 1, "records": []})
        self.assertEqual(self.archive.records_as_of(**{**self.query, "cutoff": T2})["records"], [])

    def test_publication_and_edit_times_gate_visibility(self):
        self.ingest(reports(), source_published_at=T2)
        with self.assertRaises(DataUnavailable):
            self.archive.records_as_of(**self.query)
        document = reports()
        document["records"][0]["source_updated_at"] = T2
        self.ingest(document)
        with self.assertRaises(DataUnavailable):
            self.archive.records_as_of(**self.query)

    def test_utc_equivalence_and_naive_timestamp_rejection(self):
        self.assertEqual(utc("2025-01-01T16:00:00-08:00"), utc(T1))
        with self.assertRaises(ValueError):
            utc("2025-01-01T16:00:00")

    def test_new_bad_feed_never_silently_falls_back(self):
        self.ingest(reports())
        self.time = T2
        bad = self.ingest({"schema_version": 1, "records": [{"broken": True}]})
        self.assertIsNotNone(self.archive.metadata(bad)["normalization_error"])
        with self.assertRaises(DataUnavailable):
            self.archive.records_as_of(**{**self.query, "cutoff": T2})

    def test_classification_and_freshness_gates(self):
        for data_class in ("test", "scrambled", "unverified"):
            self.ingest(reports(), data_class=data_class)
            with self.assertRaises(DataUnavailable):
                self.archive.records_as_of(**self.query)
        self.ingest(reports())
        with self.assertRaises(DataUnavailable):
            self.archive.records_as_of(**{**self.query, "cutoff": T2}, max_age_seconds=60)

    def test_raw_bytes_preserved_and_tables_immutable(self):
        raw = b' {"schema_version":1, "records":[]}\n'
        sid = self.archive.ingest(raw, source="TEST", feed=GENERIC, scope="test")
        self.assertEqual(self.archive.raw(sid), raw)
        with self.assertRaises(sqlite3.IntegrityError):
            self.archive.db.execute("UPDATE snapshots SET data_class='real' WHERE id=?", (sid,))
        with self.assertRaises(sqlite3.IntegrityError):
            self.archive.db.execute("DELETE FROM snapshots WHERE id=?", (sid,))

    def test_invalid_json_and_duplicate_records_are_quarantined_atomically(self):
        raw = b"<html>feed unavailable</html>"
        sid = self.archive.ingest(raw, source="TEST", feed=GENERIC, scope="test")
        self.assertEqual(self.archive.raw(sid), raw)
        self.assertIsNotNone(self.archive.metadata(sid)["normalization_error"])
        document = reports()
        document["records"].append(copy.deepcopy(document["records"][0]))
        sid = self.ingest(document)
        self.assertEqual(self.archive.db.execute("SELECT COUNT(*) FROM records WHERE snapshot_id=?", (sid,)).fetchone()[0], 0)

    def test_status_labels_survive_export(self):
        document = reports()
        document["records"][0]["status"] = "estimated"
        self.ingest(document)
        self.assertEqual(self.archive.records_as_of(**self.query)["records"][0]["status"], "estimated")

    def test_nonfinite_values_and_malformed_record_are_quarantined(self):
        document = reports()
        document["records"][0]["data"]["bad"] = float("nan")
        sid = self.ingest(document)
        self.assertIsNotNone(self.archive.metadata(sid)["normalization_error"])
        sid = self.ingest({"schema_version": 1, "records": [None]})
        self.assertIsNotNone(self.archive.metadata(sid)["normalization_error"])


class ReconciliationTests(unittest.TestCase):
    def evaluate(self, box, pbp):
        with Archive(":memory:", clock=lambda: T1) as archive:
            for feed, doc in ((BOX, box), (PBP, pbp)):
                archive.ingest(json.dumps(doc).encode(), source="nba", feed=feed,
                               scope=GAME_ID, data_class="real")
            return reconcile_game(archive, GAME_ID, T1)

    def test_complete_synthetic_ledger_reconciles(self):
        result = self.evaluate(*full_game())
        self.assertTrue(result["passed"], result["issues"])
        self.assertEqual(sum(result["lineups"]["player_seconds"].values()), 2 * 240 * 60)

    def test_overtime_minutes(self):
        result = self.evaluate(*full_game(overtime=True))
        self.assertTrue(result["passed"], result["issues"])
        self.assertEqual(sum(result["lineups"]["player_seconds"].values()), 2 * 265 * 60)

    def test_scoring_mismatch_fails_gate(self):
        box, pbp = full_game()
        box["game"]["homeTeam"]["players"][0]["statistics"]["points"] += 1
        result = self.evaluate(box, pbp)
        self.assertFalse(result["passed"])
        self.assertTrue(any("points" in issue for issue in result["issues"]))

    def test_missing_starters_are_not_filled(self):
        box, pbp = full_game()
        box["game"]["homeTeam"]["players"][0]["starter"] = "0"
        result = self.evaluate(box, pbp)
        self.assertFalse(result["passed"])
        self.assertEqual(result["lineups"]["status"], "unresolved")

    def test_missing_later_period_evidence_stays_unknown(self):
        box, pbp = full_game()
        pbp["game"]["actions"] = [a for a in pbp["game"]["actions"]
                                  if not (a["period"] == 2 and a["personId"] == 5)]
        result = self.evaluate(box, pbp)
        self.assertFalse(result["passed"])
        self.assertTrue(any("provable starters" in issue for issue in result["issues"]))

    def test_incomplete_or_duplicate_events_fail(self):
        box, pbp = full_game()
        pbp["game"]["actions"].pop()
        self.assertFalse(self.evaluate(box, pbp)["passed"])
        box, pbp = full_game()
        pbp["game"]["actions"].append(copy.deepcopy(pbp["game"]["actions"][0]))
        with self.assertRaises(DataUnavailable):
            self.evaluate(box, pbp)

    def test_foreign_game_id_quarantined(self):
        box, pbp = full_game()
        pbp["game"]["gameId"] = "0020000002"
        with self.assertRaises(DataUnavailable):
            self.evaluate(box, pbp)

    def test_live_game_cannot_pass_final_game_gate(self):
        box, pbp = full_game()
        box["game"]["gameStatus"] = 2
        self.assertFalse(self.evaluate(box, pbp)["passed"])

    def test_missing_period_start_cannot_pass(self):
        box, pbp = full_game()
        pbp["game"]["actions"].pop(0)
        self.assertFalse(self.evaluate(box, pbp)["passed"])

    def test_atomic_substitution_batch_accumulates_minutes(self):
        box, pbp = full_game()
        home = box["game"]["homeTeam"]
        replacement = copy.deepcopy(home["players"][0])
        replacement.update(personId=11, starter="0", name="TEST REPLACEMENT")
        replacement["statistics"] = {stat: 0 for stat in replacement["statistics"]}
        replacement["statistics"]["minutes"] = "PT10M00S"
        home["players"].append(replacement)
        # Starter 5 plays the last two minutes of Q1 plus the other three periods.
        home["players"][4]["statistics"]["minutes"] = "PT38M00S"
        events = pbp["game"]["actions"]
        for number, pid, subtype in ((900, 5, "out"), (901, 11, "in"), (902, 11, "out"), (903, 5, "in")):
            events.append({"actionNumber": number, "orderNumber": 15000 + number if number < 902 else 25000 + number,
                           "period": 1, "clock": "PT12M00S" if number < 902 else "PT02M00S",
                           "actionType": "substitution", "subType": subtype, "personId": pid,
                           "teamId": 100, "scoreHome": "0" if number < 902 else "2", "scoreAway": "0"})
        result = self.evaluate(box, pbp)
        self.assertTrue(result["passed"], result["issues"])
        self.assertEqual(result["lineups"]["player_seconds"]["11"], 600)

    def test_unpaired_substitution_never_fills_with_guessed_player(self):
        box, pbp = full_game()
        pbp["game"]["actions"].append({"actionNumber": 900, "orderNumber": 15000,
            "period": 1, "clock": "PT10M00S", "actionType": "substitution", "subType": "out",
            "personId": 5, "teamId": 100, "scoreHome": "0", "scoreAway": "0"})
        result = self.evaluate(box, pbp)
        self.assertFalse(result["passed"])
        self.assertTrue(any("substitution batch" in issue for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
