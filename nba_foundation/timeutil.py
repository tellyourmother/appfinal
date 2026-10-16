"""Strict timestamps and NBA duration parsing; never assume a local timezone."""

from datetime import datetime, timezone
import math
import re


def utc(value: str | datetime) -> str:
    value = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include Z or an explicit UTC offset")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def now() -> str:
    return utc(datetime.now(timezone.utc))


def seconds(value: str) -> float:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", value)
    if not match or not any(match.groups()):
        raise ValueError(f"Invalid NBA duration: {value!r}")
    hours, minutes, secs = (float(part or 0) for part in match.groups())
    result = hours * 3600 + minutes * 60 + secs
    if not math.isfinite(result):
        raise ValueError("Duration must be finite")
    return result


def period_seconds(period: int) -> int:
    if type(period) is not int or period < 1:
        raise ValueError("Period must be a positive integer")
    return 720 if period <= 4 else 300
