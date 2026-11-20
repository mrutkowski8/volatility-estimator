"""Estimate annualized volatility from local daily closing prices."""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory, mkdtemp
from zoneinfo import ZoneInfo

from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import pandas_market_calendars as market_calendars


DEFAULT_START = date(2026, 1, 1)
DEFAULT_END = date(2026, 10, 30)
DEFAULT_EVENTS = Path("data/events.csv")
DEFAULT_DECAY = 0.94
DEFAULT_EVENT_WINDOW = 5
CALENDAR = "NYSE"
ANNUAL_SESSIONS = 252
ROLLING_WINDOWS = (20, 60, 120)
EVENT_ZONE = "America/New_York"
EVENT_TIME_FORMAT = "%Y-%m-%d %H:%M"
FORMAT_VERSION = 1

CLOSE_COLUMNS = ("observation_date", "SP500")
EVENT_COLUMNS = ("published_et", "event_type")
STUDY_COLUMNS = (
    *EVENT_COLUMNS, "session_0", "pre_start", "pre_end", "post_start", "post_end",
    "window_sessions", "pre_ann", "post_ann", "delta_pp", "event_log_return",
)
REPORT_FILES = {"volatility.csv", "event_study.csv", "volatility.png", "run_metadata.json"}


@dataclass(frozen=True)
class Event:
    published_at: pd.Timestamp
    event_type: str


def parse_date(value):
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError(f"Invalid date {value!r}. Use YYYY-MM-DD.")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Invalid date {value!r}.") from error


def nyse_schedule(start=DEFAULT_START, end=DEFAULT_END):
    calendar = market_calendars.get_calendar(CALENDAR)
    schedule = calendar.schedule(start_date=start, end_date=end, tz="UTC")
    if schedule.empty:
        raise ValueError(f"The {CALENDAR} calendar has no sessions from {start} to {end}.")
    return schedule


def read_csv(path, required_columns):
    # Validate the same bytes used to compute the checksum.
    raw = path.read_bytes()
    try:
        contents = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: CSV files must use UTF-8 encoding.") from error

    reader = csv.DictReader(io.StringIO(contents, newline=""), strict=True)
    rows = []
    try:
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or not set(required_columns).issubset(headers):
            expected = ", ".join(required_columns)
            raise ValueError(f"{path}: Expected unique column names including {expected}. Found {headers}.")
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"{path}, line {reader.line_num}: Field count does not match the header.")
            rows.append((reader.line_num, row))
    except csv.Error as error:
        raise ValueError(f"{path}, line {reader.line_num}: Invalid CSV ({error}).") from error
    return rows, sha256(raw).hexdigest()


def read_closes(path, sessions, start, end):
    rows, checksum = read_csv(path, CLOSE_COLUMNS)
    session_dates = sessions.date
    expected_dates = set(session_dates)
    seen_dates = set()
    dated_levels = []
    levels = {}

    for line, row in rows:
        try:
            day = parse_date(row["observation_date"].strip())
        except ValueError as error:
            raise ValueError(f"{path}, line {line}: {error}") from error
        raw_level = row["SP500"].strip()
        if raw_level not in ("", "."):
            dated_levels.append(day)
        if not start <= day <= end:
            continue
        if day in seen_dates:
            raise ValueError(f"{path}, line {line}: Duplicate date {day}.")
        seen_dates.add(day)
        if raw_level in ("", "."):
            continue
        try:
            level = float(raw_level)
        except ValueError as error:
            raise ValueError(f"{path}, {day}: SP500 level {raw_level!r} is not a number.") from error
        if not math.isfinite(level) or level <= 0:
            raise ValueError(f"{path}, {day}: SP500 level must be finite and positive ({raw_level!r}).")
        if day not in expected_dates:
            raise ValueError(f"{path}, {day}: SP500 level supplied for a day with no {CALENDAR} session.")
        levels[day] = level

    missing_dates = sorted(expected_dates - levels.keys())
    if missing_dates:
        missing = ", ".join(str(day) for day in missing_dates[:6])
        if len(missing_dates) > 6:
            missing += f" and {len(missing_dates) - 6} more"
        available = f"{min(dated_levels)} to {max(dated_levels)}" if dated_levels else "none"
        raise ValueError(
            f"{path}: Missing closes for sessions {missing}. "
            f"Expected range {start} to {end}. Dates with values in the file: {available}."
        )

    close = pd.Series([levels[day] for day in session_dates], index=sessions, name="close", dtype=float)
    return close, checksum


def read_events(path):
    rows, checksum = read_csv(path, EVENT_COLUMNS)
    zone = ZoneInfo(EVENT_ZONE)
    events = set()
    for line, row in rows:
        local_time = row["published_et"].strip()
        event_type = row["event_type"].strip()
        location = f"{path}, line {line}"
        if not event_type:
            raise ValueError(f"{location}: Event type is blank for release {local_time!r}.")
        try:
            timestamp = datetime.strptime(local_time, EVENT_TIME_FORMAT)
            if timestamp.strftime(EVENT_TIME_FORMAT) != local_time:
                raise ValueError(local_time)
        except ValueError as error:
            raise ValueError(
                f"{location}: Invalid date or time {local_time!r}. Use YYYY-MM-DD HH:MM."
            ) from error
        try:
            published_at = pd.Timestamp(timestamp).tz_localize(zone, ambiguous="raise", nonexistent="raise")
        except ValueError as error:
            raise ValueError(f"{location}: Ambiguous or nonexistent local time {local_time!r} in {EVENT_ZONE}.") from error

        event = Event(published_at, event_type)
        if event in events:
            raise ValueError(f"{location}: Duplicate event {local_time} {event_type}.")
        events.add(event)
    return sorted(events, key=lambda event: (event.published_at, event.event_type)), checksum


def log_returns(close):
    return np.log(close).diff().rename("log_return")


def historical_volatility(returns):
    if returns.count() < 2:
        raise ValueError("Historical volatility requires at least two returns from three closes.")
    return float(returns.std(ddof=1) * math.sqrt(ANNUAL_SESSIONS))


def rolling_volatility(returns, window):
    return returns.rolling(window=window, min_periods=window).std(ddof=1) * math.sqrt(ANNUAL_SESSIONS)


def ewma_variance(returns, decay):
    """Compute the daily variance after each close, seeded by the first squared return."""
    squared = returns.dropna().pow(2)
    if squared.empty:
        return squared.reindex(returns.index)

    observations = iter(squared)
    variance = next(observations)
    states = [variance]
    new_weight = 1 - decay
    for observation in observations:
        variance = decay * variance + new_weight * observation
        states.append(variance)
    return pd.Series(states, index=squared.index, dtype=float).reindex(returns.index)


def realized_volatility(returns):
    return math.sqrt(ANNUAL_SESSIONS * returns.pow(2).mean())


def study_events(returns, market_closes, events, window):
    rows = []
    window_events = defaultdict(list)
    for event in events:
        published_et = event.published_at.strftime(EVENT_TIME_FORMAT)
        label = f"{published_et} ET ({event.event_type})"
        # side="right" skips a close at the exact publication instant.
        session_zero = market_closes.searchsorted(event.published_at.tz_convert("UTC"), side="right")
        if session_zero == len(returns):
            last_session = returns.index[-1].date()
            raise ValueError(f"No market close after the {label} release. Last session was {last_session}.")

        first_before = session_zero - window
        past_last_after = session_zero + 1 + window
        anchor = returns.index[session_zero].date().isoformat()
        if first_before < 1 or past_last_after > len(returns):
            available_before = max(0, session_zero - 1)
            available_after = len(returns) - session_zero - 1
            raise ValueError(
                f"The {label} release maps to session 0 on {anchor} and needs {window} returns on each side. "
                f"Found {available_before} before and {available_after} after. "
                "The first session in the sample has no return."
            )

        event_span = returns.iloc[first_before:past_last_after]
        valid_returns = np.isfinite(event_span.to_numpy(dtype=float, na_value=np.nan))
        invalid_returns = event_span[~valid_returns]
        if not invalid_returns.empty:
            dates = ", ".join(invalid_returns.index.strftime("%Y-%m-%d"))
            raise ValueError(f"The {label} release is missing valid returns for sessions {dates}.")

        before = returns.iloc[first_before:session_zero]
        after = returns.iloc[session_zero + 1:past_last_after]
        pre_ann = realized_volatility(before)
        post_ann = realized_volatility(after)
        rows.append({
            "published_et": published_et,
            "event_type": event.event_type,
            "session_0": anchor,
            "pre_start": before.index[0].date().isoformat(),
            "pre_end": before.index[-1].date().isoformat(),
            "post_start": after.index[0].date().isoformat(),
            "post_end": after.index[-1].date().isoformat(),
            "window_sessions": window,
            "pre_ann": pre_ann,
            "post_ann": post_ann,
            "delta_pp": (post_ann - pre_ann) * 100,
            "event_log_return": returns.iloc[session_zero],
        })
        for session in before.index.append(after.index):
            window_events[session.date().isoformat()].append(label)

    overlaps = {day: labels for day, labels in sorted(window_events.items()) if len(labels) > 1}
    return pd.DataFrame(rows, columns=STUDY_COLUMNS), overlaps


def plot_volatility(path, daily, event_study, decay):
    figure = Figure(figsize=(11.5, 5.4), layout="constrained")
    axes = figure.subplots()
    colors = ("#23658c", "#277350", "#694e9a")
    styles = ("-", "--", "-.")
    for window, color, style in zip(ROLLING_WINDOWS, colors, styles, strict=True):
        axes.plot(daily.index, daily[f"hist_{window}_ann"], color=color, linestyle=style, label=f"{window} sessions")
    axes.plot(daily.index, daily["ewma_ann"], color="#ad4f32", linewidth=1.8, label=f"EWMA (λ={decay:g})")
    for number, session in enumerate(event_study["session_0"].drop_duplicates()):
        axes.axvline(pd.Timestamp(session), color="#687582", linestyle=":", linewidth=1,
                     label="Session 0" if number == 0 else None)

    locator = AutoDateLocator(minticks=4, maxticks=8)
    axes.xaxis.set_major_locator(locator)
    axes.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes.set(title="Daily return volatility", ylabel="Annualized volatility", xlabel="NYSE session date")
    axes.set_ylim(bottom=0)
    axes.grid(axis="y", alpha=0.22)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=5, frameon=False)
    figure.savefig(path, dpi=150, backend="agg")


def checked_output_path(output_dir, inputs):
    destination = output_dir.resolve()
    if destination == destination.parent:
        raise ValueError(f"Choose a separate output directory instead of {destination}.")
    for path in inputs:
        source = path.resolve()
        if destination == source or destination in source.parents or source in destination.parents:
            raise ValueError(f"Output directory {destination} conflicts with input file {source}.")
    return destination


def replace_report(staged, destination):
    # Keep the previous bundle available for rollback and serialize concurrent writers.
    lock = destination.with_name(f".{destination.name}.lock")
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise ValueError(f"Output directory is busy. Check {lock}.") from error

    previous = None
    try:
        if destination.exists():
            if not destination.is_dir():
                raise ValueError(f"Output path {destination} is a file.")
            for entry in destination.iterdir():
                if entry.name not in REPORT_FILES or not entry.is_file() or entry.is_symlink():
                    raise ValueError(f"Directory {destination} contains {entry.name!r}. Choose a different output directory.")
            backup = Path(mkdtemp(dir=destination.parent, prefix=".volatility-previous-"))
            previous = backup / "report"
            destination.rename(previous)

        try:
            staged.rename(destination)
        except OSError as error:
            if previous is not None:
                try:
                    previous.rename(destination)
                except OSError as restore_error:
                    raise OSError(f"Could not restore the previous report ({restore_error}). It is saved at {previous}.") from error
            raise

        if previous is not None:
            try:
                for entry in previous.iterdir():
                    entry.unlink()
                previous.rmdir()
            except OSError as error:
                print(
                    f"The new report is ready. Could not remove the previous report at {previous} ({error}).",
                    file=sys.stderr,
                )
    finally:
        try:
            if previous is not None and not previous.exists():
                previous.parent.rmdir()
        finally:
            lock.rmdir()


def write_report(destination, daily, event_study, metadata):
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent, prefix=".volatility-") as temporary:
            staged = Path(temporary)
            daily.to_csv(staged / "volatility.csv", index_label="date", date_format="%Y-%m-%d")
            event_study.to_csv(staged / "event_study.csv", index=False)
            plot_volatility(staged / "volatility.png", daily, event_study, metadata["decay"])
            (staged / "run_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            replace_report(staged, destination)
    except OSError as error:
        raise OSError(f"Could not save the report to {destination} ({error}).") from error


def run(args):
    output_dir = args.output_dir if args.output_dir else Path("output") / args.closes.stem
    destination = checked_output_path(output_dir, (args.closes, args.events))
    schedule = nyse_schedule(args.start, args.end)
    close, close_hash = read_closes(args.closes, schedule.index, args.start, args.end)
    events, events_hash = read_events(args.events)

    returns = log_returns(close)
    full_sample = historical_volatility(returns)
    daily = pd.DataFrame({"close": close, "log_return": returns})
    for window in ROLLING_WINDOWS:
        daily[f"hist_{window}_ann"] = rolling_volatility(returns, window)
    daily["ewma_ann"] = np.sqrt(ANNUAL_SESSIONS * ewma_variance(returns, args.decay))
    event_study, overlaps = study_events(returns, schedule["market_close"], events, args.event_window)

    metadata = {
        "format_version": FORMAT_VERSION,
        "source_label": args.source_label,
        "inputs": {
            "closes": {"filename": args.closes.name, "sha256": close_hash},
            "events": {"filename": args.events.name, "sha256": events_hash},
        },
        "calendar": CALENDAR,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "first_session": schedule.index[0].date().isoformat(),
        "last_session": schedule.index[-1].date().isoformat(),
        "event_timezone": EVENT_ZONE,
        "annualization_sessions": ANNUAL_SESSIONS,
        "rolling_windows": list(ROLLING_WINDOWS),
        "decay": args.decay,
        "event_window": args.event_window,
        "session_count": len(close),
        "return_count": int(returns.count()),
        "event_count": len(event_study),
        "historical_volatility_ann": full_sample,
        "overlapping_window_sessions": overlaps,
    }
    write_report(destination, daily, event_study, metadata)
    print(f"Processed {metadata['session_count']} closes, {metadata['return_count']} returns, and {len(event_study)} events.")
    print(f"Historical volatility over the full sample is {full_sample:.2%} annualized.")
    if overlaps:
        print(f"Event windows share these sessions: {', '.join(overlaps)}. See run_metadata.json for details.")
    print(f"Results saved to {destination}")


def cli_date(value):
    try:
        return parse_date(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estimate S&P 500 volatility and compare windows around economic releases.")
    parser.add_argument("closes", type=Path, help="Local CSV file with observation_date and SP500 columns.")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS, help=f"CSV of releases. Defaults to {DEFAULT_EVENTS.as_posix()}.")
    parser.add_argument("--start", type=cli_date, default=DEFAULT_START, help=f"Start date (inclusive). Defaults to {DEFAULT_START}.")
    parser.add_argument("--end", type=cli_date, default=DEFAULT_END, help=f"End date (inclusive). Defaults to {DEFAULT_END}.")
    parser.add_argument("--event-window", type=int, default=DEFAULT_EVENT_WINDOW,
                        help=f"Returns on each side of session 0. Defaults to {DEFAULT_EVENT_WINDOW}.")
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY,
                        help=f"Weight on the previous EWMA state, between 0 and 1. Defaults to {DEFAULT_DECAY}.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to output/<input filename without extension>.")
    parser.add_argument("--source-label", default="local-file", help="Source name to record in metadata. Defaults to local-file.")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must be on or before --end.")
    if args.event_window < 1:
        parser.error("--event-window must be a positive number of sessions.")
    if not 0 < args.decay < 1:
        parser.error("--decay must be finite and strictly between 0 and 1.")
    if not args.source_label.strip():
        parser.error("--source-label must contain a source name.")
    try:
        run(args)
    except (OSError, ValueError, OverflowError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


def configure_terminal():
    # Preserve Unicode in paths when output is redirected on Windows.
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


if __name__ == "__main__":
    configure_terminal()
    raise SystemExit(main())
