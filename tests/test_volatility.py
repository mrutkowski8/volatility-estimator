import csv
from datetime import date
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

import volatility as vol


PROJECT = Path(__file__).resolve().parents[1]


class VolatilityTests(unittest.TestCase):
    def test_log_returns_and_scale_invariance(self):
        levels = pd.Series([100, 110, 99], dtype=float)
        returns = vol.log_returns(levels)
        expected = [np.nan, math.log(1.1), math.log(0.9)]
        np.testing.assert_allclose(returns, expected, atol=1e-14)
        np.testing.assert_allclose(vol.log_returns(levels * 3), returns, atol=1e-14)
        self.assertAlmostEqual(returns.sum(), math.log(0.99))

    def test_sample_variance_uses_n_minus_one(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        self.assertAlmostEqual(vol.historical_volatility(returns), math.sqrt(252 * 0.0001))
        with self.assertRaisesRegex(ValueError, "at least two returns"):
            vol.historical_volatility(returns.iloc[:1])

    def test_equal_nonzero_returns_have_realized_volatility(self):
        returns = pd.Series([0.01, 0.01, 0.01])
        self.assertAlmostEqual(vol.historical_volatility(returns), 0.0)
        self.assertAlmostEqual(vol.realized_volatility(returns), 0.01 * math.sqrt(252))

    def test_full_windows_begin_after_the_initial_missing_return(self):
        returns = vol.log_returns(pd.Series([100.0] * 125))
        for window in (20, 60, 120):
            with self.subTest(window=window):
                series = vol.rolling_volatility(returns, window)
                self.assertEqual(series.first_valid_index(), window)
                self.assertTrue(series.iloc[:window].isna().all())
                self.assertTrue((series.iloc[window:] == 0).all())
        self.assertTrue(vol.rolling_volatility(returns.iloc[:100], 120).isna().all())

    def test_ewma_shock_and_seed(self):
        returns = pd.Series([np.nan, 0.01, -0.04, 0.0])
        actual = vol.ewma_variance(returns, 0.94)
        reference = returns.dropna().pow(2).ewm(alpha=0.06, adjust=False).mean().reindex(returns.index)
        np.testing.assert_allclose(actual, [np.nan, 0.0001, 0.000190, 0.0001786], atol=1e-15)
        pd.testing.assert_series_equal(actual, reference)

    def test_latest_close_does_not_change_earlier_estimates(self):
        levels = pd.Series(np.linspace(100.0, 125.0, num=125))
        returns = vol.log_returns(levels)
        levels.iloc[-1] *= 0.85
        changed_returns = vol.log_returns(levels)
        for window in (20, 60, 120):
            pd.testing.assert_series_equal(
                vol.rolling_volatility(returns, window).iloc[:-1],
                vol.rolling_volatility(changed_returns, window).iloc[:-1],
            )
        pd.testing.assert_series_equal(vol.ewma_variance(returns, 0.94).iloc[:-1],
                                       vol.ewma_variance(changed_returns, 0.94).iloc[:-1])
        self.assertNotAlmostEqual(vol.historical_volatility(returns), vol.historical_volatility(changed_returns))


class EventStudyTests(unittest.TestCase):
    def event(self, local_time, event_type="CPI"):
        return vol.Event(pd.Timestamp(local_time, tz="America/New_York"), event_type)

    def test_after_friday_close_and_exact_close_map_to_monday(self):
        schedule = vol.nyse_schedule(date(2026, 1, 2), date(2026, 1, 16))
        returns = vol.log_returns(pd.Series(np.arange(100.0, 100.0 + len(schedule)), index=schedule.index))
        for timestamp in ("2026-01-09 16:00", "2026-01-09 17:00", "2026-01-10 08:30"):
            with self.subTest(timestamp=timestamp):
                study, _ = vol.study_events(returns, schedule.market_close, [self.event(timestamp)], 2)
                row = study.iloc[0]
                self.assertEqual(row.session_0, "2026-01-12")
                self.assertEqual((row.pre_start, row.pre_end), ("2026-01-08", "2026-01-09"))
                self.assertEqual((row.post_start, row.post_end), ("2026-01-13", "2026-01-14"))

    def test_early_close_in_the_exchange_calendar_is_used(self):
        schedule = vol.nyse_schedule(date(2026, 11, 23), date(2026, 12, 2))
        self.assertEqual(schedule.loc["2026-11-27", "market_close"], pd.Timestamp("2026-11-27 18:00Z"))
        returns = vol.log_returns(pd.Series(np.arange(100.0, 100.0 + len(schedule)), index=schedule.index))
        events = [self.event("2026-11-27 12:59"), self.event("2026-11-27 13:00")]
        study, _ = vol.study_events(returns, schedule.market_close, events, 1)
        self.assertEqual(study.session_0.tolist(), ["2026-11-27", "2026-11-30"])

    def test_event_return_is_excluded_from_both_windows(self):
        schedule = vol.nyse_schedule(date(2026, 1, 2), date(2026, 1, 9))
        returns = pd.Series([np.nan, -0.01, 0.01, 0.05, 0.02, -0.02], index=schedule.index)
        events = [self.event("2026-01-07 08:30")]
        study, _ = vol.study_events(returns, schedule.market_close, events, 2)
        row = study.iloc[0]
        self.assertAlmostEqual(row.pre_ann, 0.01 * math.sqrt(252))
        self.assertAlmostEqual(row.post_ann, 0.02 * math.sqrt(252))
        self.assertAlmostEqual(row.delta_pp, 0.01 * math.sqrt(252) * 100)
        self.assertAlmostEqual(row.event_log_return, 0.05)
        returns.iloc[3] = -0.04
        changed, _ = vol.study_events(returns, schedule.market_close, events, 2)
        self.assertAlmostEqual(changed.iloc[0].event_log_return, -0.04)
        pd.testing.assert_frame_equal(study.drop(columns="event_log_return"), changed.drop(columns="event_log_return"))

    def test_overlaps_list_only_sessions_in_both_windows(self):
        schedule = vol.nyse_schedule(date(2026, 1, 2), date(2026, 1, 16))
        returns = vol.log_returns(pd.Series(np.arange(100.0, 100.0 + len(schedule)), index=schedule.index))
        events = [self.event("2026-01-08 08:30"), self.event("2026-01-09 14:00", "FOMC")]
        _, overlaps = vol.study_events(returns, schedule.market_close, events, 2)
        self.assertEqual(list(overlaps), ["2026-01-07", "2026-01-12"])
        self.assertTrue(all(len(labels) == 2 for labels in overlaps.values()))

    def test_incomplete_windows_and_event_after_sample_report_publication(self):
        schedule = vol.nyse_schedule(date(2026, 1, 2), date(2026, 1, 9))
        returns = vol.log_returns(pd.Series(np.arange(100.0, 100.0 + len(schedule)), index=schedule.index))
        for timestamp, expected in (
            ("2026-01-06 08:30", r"2026-01-06.*Found 1 before"),
            ("2026-01-08 08:30", r"2026-01-08.*and 1 after"),
            ("2026-01-09 16:00", r"No market close after the 2026-01-09"),
        ):
            with self.subTest(timestamp=timestamp), self.assertRaisesRegex(ValueError, expected):
                vol.study_events(returns, schedule.market_close, [self.event(timestamp)], 2)

    def test_event_windows_require_finite_returns_for_every_session(self):
        schedule = vol.nyse_schedule(date(2026, 1, 2), date(2026, 1, 9))
        events = [self.event("2026-01-07 08:30")]
        for position, invalid, dtype in ((1, np.nan, float), (3, pd.NA, "Float64"), (5, np.inf, float)):
            returns = pd.Series([np.nan, -0.01, 0.01, 0.05, 0.02, -0.02], index=schedule.index, dtype=dtype)
            returns.iloc[position] = invalid
            day = returns.index[position].date()
            with self.subTest(day=day), self.assertRaisesRegex(ValueError, rf"2026-01-07.*sessions {day}"):
                vol.study_events(returns, schedule.market_close, events, 2)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.start = date(2026, 1, 1)
        self.end = date(2026, 1, 7)
        self.sessions = vol.nyse_schedule(self.start, self.end).index
        self.valid = "2026-01-07,103\n2026-01-02,100\n2026-01-05,101\n2026-01-06,102\n"
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input.csv"

    def closes(self, rows):
        self.path.write_text("observation_date,SP500\n" + rows, encoding="utf-8")
        return vol.read_closes(self.path, self.sessions, self.start, self.end)

    def events(self, rows):
        self.path.write_text("published_et,event_type\n" + rows, encoding="utf-8")
        return vol.read_events(self.path)

    def test_unsorted_dates_off_days_and_explicit_range(self):
        close, checksum = self.closes(self.valid + "2026-01-01,.\n2026-01-03,\n2025-12-31,ignored\n")
        self.assertEqual(close.tolist(), [100, 101, 102, 103])
        self.assertTrue(close.index.equals(self.sessions))
        self.assertEqual(checksum, sha256(self.path.read_bytes()).hexdigest())

    def test_invalid_close_rows(self):
        cases = (
            (self.valid + "2026-01-05,105\n", "Duplicate date 2026-01-05"),
            (self.valid + "2026-01-03,105\n", "2026-01-03.*no NYSE session"),
            (self.valid + "2026-02-30,105\n", "Invalid date.*2026-02-30"),
            (self.valid.replace("2026-01-05,101", "2026-01-05,abc"), "2026-01-05.*is not a number"),
            (self.valid.replace("2026-01-05,101", "2026-01-05,0"), "2026-01-05.*finite and positive"),
            (self.valid.replace("2026-01-05,101", "2026-01-05,inf"), "2026-01-05.*finite and positive"),
            (self.valid.replace("2026-01-05,101\n", ""), "Missing closes.*2026-01-05"),
            (self.valid.replace("2026-01-05,101", "2026-01-05,."), "Missing closes.*2026-01-05"),
            (self.valid.replace("2026-01-07,103\n", ""), "2026-01-07.*2026-01-02 to 2026-01-06"),
            ("2025-01-02,100\n", "2026-01-01 to 2026-01-07.*2025-01-02 to 2025-01-02"),
        )
        for rows, expected in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                self.closes(rows)

    def test_event_validation_reports_row_and_publication(self):
        cases = (
            ("2026-04-10 25:30,CPI\n", "Invalid date or time.*2026-04-10 25:30"),
            ("2026-04-10 08:30, \n", "Event type is blank.*2026-04-10 08:30"),
            ("2026-04-10 08:30,CPI\n" * 2, "line 3.*Duplicate event"),
            ("2026-03-08 02:30,CPI\n", "line 2.*nonexistent local time.*2026-03-08 02:30"),
            ("2026-11-01 01:30,CPI\n", "line 2.*Ambiguous.*2026-11-01 01:30"),
        )
        for rows, expected in cases:
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, expected):
                self.events(rows)

    def test_events_sort_chronologically_with_dst_offsets(self):
        events, _ = self.events("2026-04-10 08:30,CPI\n2026-01-09 12:30,FOMC\n")
        self.assertEqual([event.event_type for event in events], ["FOMC", "CPI"])
        self.assertEqual(events[0].published_at.tz_convert("UTC").hour, 17)
        self.assertEqual(events[1].published_at.tz_convert("UTC").hour, 12)


class OutputTests(unittest.TestCase):
    def test_input_cannot_be_used_as_output_or_contained_in_it(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "data" / "closes.csv"
            for output in (path, path.parent, path / "child"):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "conflicts with"):
                    vol.checked_output_path(output, (path,))

    def test_failed_report_write_preserves_previous_bundle(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            destination = root / "out"
            destination.mkdir()
            report = destination / "volatility.csv"
            report.write_bytes(b"original")
            with patch.object(vol, "plot_volatility", side_effect=OSError("write interrupted")):
                with self.assertRaisesRegex(OSError, "write interrupted"):
                    vol.write_report(destination, pd.DataFrame(), pd.DataFrame(), {"decay": 0.94})
            self.assertEqual(report.read_bytes(), b"original")
            self.assertEqual(list(root.iterdir()), [destination])

    def test_replacement_restores_previous_bundle_if_rename_fails(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            staged = root / "staged"
            staged.mkdir()
            destination = root / "out"
            destination.mkdir()
            original = destination / "volatility.csv"
            original.write_bytes(b"original")
            rename = Path.rename

            def rename_except_stage(path, target):
                if path == staged:
                    raise OSError("rename interrupted")
                return rename(path, target)

            with patch.object(Path, "rename", rename_except_stage):
                with self.assertRaisesRegex(OSError, "rename interrupted"):
                    vol.replace_report(staged, destination)
            self.assertEqual(original.read_bytes(), b"original")
            self.assertEqual(set(root.iterdir()), {destination, staged})

    def test_backup_cleanup_failure_does_not_block_another_report(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            staged = root / "staged"
            destination = root / "out"
            destination.mkdir()
            report = destination / "volatility.csv"
            report.write_bytes(b"original")
            staged.mkdir()
            (staged / report.name).write_bytes(b"new report")
            with (
                patch.object(Path, "unlink", side_effect=PermissionError("access denied")),
                patch.object(sys, "stderr", new_callable=io.StringIO) as warning,
            ):
                vol.replace_report(staged, destination)

            self.assertEqual(report.read_bytes(), b"new report")
            backup_files = list(root.glob(".volatility-previous-*/report/volatility.csv"))
            self.assertEqual(len(backup_files), 1)
            self.assertEqual(backup_files[0].read_bytes(), b"original")
            self.assertIn(str(backup_files[0].parent), warning.getvalue())
            self.assertFalse((root / ".out.lock").exists())

            staged.mkdir()
            (staged / report.name).write_bytes(b"next report")
            vol.replace_report(staged, destination)
            self.assertEqual(report.read_bytes(), b"next report")


class CliTests(unittest.TestCase):
    def cli(self, cwd, script, *args):
        return subprocess.run(
            [sys.executable, str(PROJECT / script), *args], cwd=cwd,
            capture_output=True, encoding="utf-8", timeout=30, check=False,
        )

    def test_invalid_parameters_fail_before_reading_inputs(self):
        with TemporaryDirectory() as temp:
            for args, expected in (
                (("--event-window", "0"), "a positive number of sessions"),
                (("--decay", "nan"), "strictly between 0 and 1"),
                (("--decay", "1"), "strictly between 0 and 1"),
                (("--start", "20260102"), "Invalid date"),
                (("--start", "2026-11-01"), "on or before --end"),
            ):
                with self.subTest(args=args):
                    result = self.cli(temp, "volatility.py", "missing.csv", *args)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stderr)
                    self.assertFalse((Path(temp) / "output").exists())

    def test_demo_cli_exports_complete_and_reproducible_results(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            generated = self.cli(root, "make_demo.py")
            self.assertEqual(generated.returncode, 0, generated.stderr)
            prices = root / "data/demo.csv"
            price_bytes = prices.read_bytes()
            shutil.copyfile(PROJECT / "data/events.csv", root / "data/events.csv")
            args = ("data/demo.csv", "--source-label", "synthetic-demo")
            result = self.cli(root, "volatility.py", *args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("209", result.stdout)
            destination = root / "output/demo"
            self.assertEqual({path.name for path in destination.iterdir()}, {
                "volatility.csv", "event_study.csv", "volatility.png", "run_metadata.json",
            })

            with (destination / "volatility.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                daily = list(reader)
                self.assertEqual(reader.fieldnames, [
                    "date", "close", "log_return", "hist_20_ann", "hist_60_ann", "hist_120_ann", "ewma_ann",
                ])
            self.assertEqual(len(daily), 209)
            self.assertEqual((daily[0]["date"], daily[0]["log_return"], daily[0]["ewma_ann"]), ("2026-01-02", "", ""))
            for window, expected in ((20, "2026-02-02"), (60, "2026-03-31"), (120, "2026-06-26")):
                column = f"hist_{window}_ann"
                self.assertEqual(next(row["date"] for row in daily if row[column]), expected)

            with (destination / "event_study.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                study = list(reader)
                self.assertEqual(reader.fieldnames, list(vol.STUDY_COLUMNS))
            self.assertEqual(len(study), 4)
            shock = study[-1]
            self.assertEqual(shock["session_0"], "2026-08-12")
            self.assertEqual((shock["pre_start"], shock["pre_end"]), ("2026-08-05", "2026-08-11"))
            self.assertEqual((shock["post_start"], shock["post_end"]), ("2026-08-13", "2026-08-19"))
            self.assertAlmostEqual(float(shock["event_log_return"]), -0.04)
            self.assertAlmostEqual(float(shock["delta_pp"]), 0.0)

            with Image.open(destination / "volatility.png") as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 1000)
                darkest, lightest = image.convert("L").getextrema()
                self.assertLess(darkest, lightest)

            metadata_file = destination / "run_metadata.json"
            meta = json.loads(metadata_file.read_text(encoding="utf-8"))
            self.assertEqual((meta["session_count"], meta["return_count"], meta["event_count"]), (209, 208, 4))
            self.assertEqual(meta["source_label"], "synthetic-demo")
            self.assertEqual(meta["inputs"]["closes"]["sha256"], sha256(price_bytes).hexdigest())
            self.assertEqual(meta["format_version"], 1)

            second = self.cli(root, "volatility.py", *args, "--event-window", "10")
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(json.loads(metadata_file.read_text(encoding="utf-8"))["event_window"], 10)
            self.assertEqual(prices.read_bytes(), price_bytes)
            self.assertEqual(list(destination.parent.iterdir()), [destination])
            regenerated = self.cli(root, "make_demo.py")
            self.assertEqual(regenerated.returncode, 0, regenerated.stderr)
            self.assertEqual(prices.read_bytes(), price_bytes)


if __name__ == "__main__":
    unittest.main()
