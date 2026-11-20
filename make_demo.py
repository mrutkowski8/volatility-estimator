"""Generate deterministic practice prices for the reference NYSE sessions."""

import csv
from datetime import date
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from volatility import CLOSE_COLUMNS, configure_terminal, nyse_schedule


def write_demo(path, *, include_shock=True):
    sessions = nyse_schedule().index.date
    shock_session = date(2026, 8, 12)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".demo-") as directory:
        staged = Path(directory) / path.name
        with staged.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(CLOSE_COLUMNS)
            writer.writerow((sessions[0], 100))
            cumulative_return = 0.0
            for position, session in enumerate(sessions[1:], start=1):
                daily_return = -0.003 if position % 2 else 0.003
                if include_shock and session == shock_session:
                    daily_return = -0.04
                cumulative_return += daily_return
                writer.writerow((session, 100 * math.exp(cumulative_return)))
        staged.replace(path)
    return len(sessions)


if __name__ == "__main__":
    configure_terminal()
    demo_path = Path("data/demo.csv")
    try:
        session_count = write_demo(demo_path)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not save {demo_path} ({error}).") from error
    print(f"Wrote {session_count} closing prices to {demo_path}")
