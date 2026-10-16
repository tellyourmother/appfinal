from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from nba_foundation.__main__ import main
from nba_foundation.store import Archive
from tests.fixtures import T0, T2, reports


class CLITests(unittest.TestCase):
    def invoke(self, *args):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_disk_archive_survives_reopen_and_refuses_backdating(self):
        with TemporaryDirectory() as directory:
            db = str(Path(directory) / "archive.sqlite3")
            payload = Path(directory) / "test.json"
            payload.write_text(json.dumps(reports()))
            code, out, err = self.invoke("--db", db, "ingest", str(payload), "--source", "TEST",
                "--feed", "records.v1", "--scope", "daily", "--source-uri", "test://synthetic",
                "--data-class", "test", "--source-published-at", T0)
            self.assertEqual(code, 0, err)
            sid = json.loads(out)["id"]
            with Archive(db) as archive:
                self.assertEqual(archive.raw(sid), payload.read_bytes())
                self.assertGreater(archive.metadata(sid)["retrieved_at"], T2)
            code, out, err = self.invoke("--db", db, "query", "--source", "TEST", "--feed", "records.v1",
                                         "--scope", "daily", "--cutoff", T0)
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(err)["status"], "withheld")

    def test_fetch_failure_is_explicit_and_creates_no_fake_input(self):
        with TemporaryDirectory() as directory:
            db = str(Path(directory) / "archive.sqlite3")
            with patch("nba_foundation.fetch.urlopen", side_effect=HTTPError("test://nba", 403, "Forbidden", {}, None)):
                code, out, err = self.invoke("--db", db, "fetch-game", "0022000001")
            self.assertEqual(code, 2)
            self.assertIn("403", json.loads(err)["error"])
            with Archive(db) as archive:
                self.assertEqual(archive.db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0], 0)

    def test_malformed_ingest_keeps_raw_response_and_exits_nonzero(self):
        with TemporaryDirectory() as directory:
            db = str(Path(directory) / "archive.sqlite3")
            payload = Path(directory) / "bad.json"
            payload.write_bytes(b"invalid JSON")
            code, out, err = self.invoke("--db", db, "ingest", str(payload), "--source", "TEST",
                "--feed", "records.v1", "--scope", "daily", "--source-uri", "test://invalid")
            self.assertEqual(code, 2)
            self.assertIn("normalization_error", json.loads(out))
