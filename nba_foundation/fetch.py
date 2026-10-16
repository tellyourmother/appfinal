"""Bounded downloads from the official NBA Live endpoint family."""

import re
from urllib.request import Request, urlopen

from .adapters import BOX, PBP

MAX_BYTES = 16 * 1024 * 1024


def fetch_game(archive, game_id: str) -> list[dict]:
    if not re.fullmatch(r"\d{10}", game_id):
        raise ValueError("NBA game ID must contain ten digits")
    snapshots = []
    for endpoint, feed in (("boxscore", BOX), ("playbyplay", PBP)):
        url = f"https://cdn.nba.com/static/json/liveData/{endpoint}/{endpoint}_{game_id}.json"
        request = Request(url, headers={"User-Agent": "nba-foundation/0.1", "Accept": "application/json"})
        with urlopen(request, timeout=30) as response:
            raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("NBA response exceeds 16 MiB limit")
            # Keep useful response provenance, never authorization/cookie headers.
            headers = {key: response.headers[key] for key in
                       ("Date", "Last-Modified", "ETag", "Content-Type") if key in response.headers}
        snapshot_id = archive.ingest(raw, source="nba", feed=feed, scope=game_id,
                                     data_class="real", source_uri=url, response_headers=headers)
        result = archive.metadata(snapshot_id)
        snapshots.append(result)
        if result["normalization_error"]:
            raise ValueError(f"Snapshot {snapshot_id} quarantined: {result['normalization_error']}")
    return snapshots
