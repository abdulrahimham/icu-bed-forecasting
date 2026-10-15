"""Shared backtesting and scoring code, so every forecasting approach is scored the same way.

A forecast "rule" is any function rule(origin_week, target_week, history) that returns a
forecast of the share of ICU beds in use (in percent) for target_week, using only `history`:
the weekly data up to and including origin_week.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DATA_FOLDER = Path(__file__).resolve().parent.parent / "data" / "processed"

FORECAST_HORIZONS = [1, 2, 3, 4]
TEST_PERIOD_START = pd.Timestamp("2024-11-09")
VALIDATION_PERIOD_END = pd.Timestamp("2026-03-28")
HOLDOUT_PERIOD_START = pd.Timestamp("2026-04-04")  # scored only in the final evaluation
RANGE_COVERAGE_TARGET = 0.80
MINIMUM_PAST_ERRORS_FOR_RANGE = 26


def load_icu_weekly():
    icu_weekly = pd.read_csv(PROCESSED_DATA_FOLDER / "icu_weekly_california.csv", parse_dates=["week_ending"])
    return icu_weekly.sort_values("week_ending").reset_index(drop=True)


def run_backtest(icu_weekly, forecast_rule, rule_name, range_errors_since=None):
    """Forecast 1 to 4 weeks ahead from every week, using only data available at that week.

    range_errors_since: if set, the 80% range is built only from past errors whose target week
    is on or after this date (for example, to leave out the large swings of 2021 and 2022).
    """
    all_weeks = icu_weekly["week_ending"]
    share_by_week = icu_weekly.set_index("week_ending")["icu_occupancy_share_pct"]
    is_flagged_by_week = icu_weekly.set_index("week_ending")["is_voluntary_reporting_week"]
    beds_available_by_week = icu_weekly.set_index("week_ending")["icu_beds_available"]
    beds_occupied_by_week = icu_weekly.set_index("week_ending")["icu_beds_occupied"]

    rows = []
    for origin_index, origin_week in enumerate(all_weeks):
        history = icu_weekly.iloc[: origin_index + 1]
        for horizon in FORECAST_HORIZONS:
            target_index = origin_index + horizon
            if target_index >= len(all_weeks):
                continue
            target_week = all_weeks[target_index]
            rows.append({
                "rule": rule_name,
                "origin_week": origin_week,
                "horizon_weeks": horizon,
                "target_week": target_week,
                "forecast_share_pct": forecast_rule(origin_week, target_week, history),
                "actual_share_pct": share_by_week[target_week],
                "usable": not (is_flagged_by_week[origin_week] or is_flagged_by_week[target_week]),
            })
    backtest = pd.DataFrame(rows)
    backtest["error_share_pts"] = backtest["actual_share_pct"] - backtest["forecast_share_pct"]

    # 80% range from this rule's own past errors at the same horizon, using only target weeks
    # that had already happened by the forecast origin
    lower_quantile = (1 - RANGE_COVERAGE_TARGET) / 2
    upper_quantile = 1 - lower_quantile
    error_pool = backtest[backtest["usable"] & backtest["error_share_pts"].notna()]
    if range_errors_since is not None:
        error_pool = error_pool[error_pool["target_week"] >= pd.Timestamp(range_errors_since)]
    lower_bounds, upper_bounds = [], []
    for row in backtest.itertuples():
        past_errors = error_pool.loc[
            (error_pool["horizon_weeks"] == row.horizon_weeks) & (error_pool["target_week"] <= row.origin_week),
            "error_share_pts",
        ]
        if len(past_errors) < MINIMUM_PAST_ERRORS_FOR_RANGE:
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
            continue
        lower_bounds.append(row.forecast_share_pct + past_errors.quantile(lower_quantile))
        upper_bounds.append(row.forecast_share_pct + past_errors.quantile(upper_quantile))
    backtest["range_low_pct"] = lower_bounds
    backtest["range_high_pct"] = upper_bounds

    # Convert to patients using the average bed count over the 4 weeks up to the origin
    recent_average_beds = beds_available_by_week.rolling(4).mean()
    backtest["beds_used_for_conversion"] = backtest["origin_week"].map(recent_average_beds)
    backtest["forecast_patients"] = backtest["forecast_share_pct"] / 100 * backtest["beds_used_for_conversion"]
    backtest["actual_patients"] = backtest["target_week"].map(beds_occupied_by_week)
    backtest["error_patients"] = backtest["actual_patients"] - backtest["forecast_patients"]

    backtest["period"] = np.select(
        [backtest["target_week"] < TEST_PERIOD_START, backtest["target_week"] <= VALIDATION_PERIOD_END],
        ["Before test period", "Validation"],
        default="Holdout",
    )
    return backtest


def validation_results(backtest):
    """Only usable validation forecasts. Holdout forecasts are never returned here."""
    return backtest[(backtest["period"] == "Validation") & backtest["usable"]].copy()


def summarize(results):
    inside_range = (results["actual_share_pct"] >= results["range_low_pct"]) & (
        results["actual_share_pct"] <= results["range_high_pct"]
    )
    return pd.Series({
        "avg_error_share_pts": results["error_share_pts"].abs().mean(),
        "avg_error_patients": results["error_patients"].abs().mean(),
        "range_coverage_pct": inside_range.mean() * 100,
        "avg_range_width_pts": (results["range_high_pct"] - results["range_low_pct"]).mean(),
    })


def score_by_horizon(results):
    return results.groupby(["rule", "horizon_weeks"]).apply(summarize, include_groups=False).round(2)


def compare_with_baselines(model_scores, baseline_scores, measure="avg_error_share_pts"):
    """For each approach and horizon: its error, the better baseline's error, and the % improvement.

    A positive "pct_better_than_baseline" means the approach beat the better of the two baselines.
    """
    best_baseline = baseline_scores[measure].groupby("horizon_weeks").min()
    comparison = model_scores[[measure]].rename(columns={measure: "approach_error"}).reset_index()
    comparison["best_baseline_error"] = comparison["horizon_weeks"].map(best_baseline)
    comparison["pct_better_than_baseline"] = (
        (comparison["best_baseline_error"] - comparison["approach_error"]) / comparison["best_baseline_error"] * 100
    )
    return comparison.set_index(["rule", "horizon_weeks"]).round(2)


def improvement_uncertainty(results, approach_rule, baseline_rule="Same as this week",
                            block_length_weeks=8, resamples=4000, seed=0):
    """How sure can we be that an approach beats a baseline? (validation results only)

    For each horizon, compares the two rules' errors week by week, then resamples the weeks in
    blocks (neighboring weeks are similar, so they are kept together) many times to see how much
    the % improvement could change by chance. If the 90% range includes 0, the improvement is
    not clear.
    """
    random_generator = np.random.default_rng(seed)
    rows = []
    for horizon, horizon_results in results.groupby("horizon_weeks"):
        errors = horizon_results.pivot_table(index="target_week", columns="rule", values="error_share_pts").dropna()
        error_difference = (errors[baseline_rule].abs() - errors[approach_rule].abs()).to_numpy()
        baseline_average_error = errors[baseline_rule].abs().mean()
        weeks = len(error_difference)
        blocks_needed = int(np.ceil(weeks / block_length_weeks))
        resampled_improvements = []
        for _ in range(resamples):
            block_starts = random_generator.integers(0, weeks - block_length_weeks + 1, size=blocks_needed)
            resampled = np.concatenate([error_difference[start:start + block_length_weeks] for start in block_starts])[:weeks]
            resampled_improvements.append(resampled.mean() / baseline_average_error * 100)
        rows.append({
            "horizon_weeks": horizon,
            "weeks_compared": weeks,
            "pct_of_weeks_approach_was_better": (error_difference > 0).mean() * 100,
            "pct_better_than_baseline": error_difference.mean() / baseline_average_error * 100,
            "likely_range_low": np.percentile(resampled_improvements, 5),
            "likely_range_high": np.percentile(resampled_improvements, 95),
        })
    table = pd.DataFrame(rows).set_index("horizon_weeks").round(1)
    table["clear_improvement"] = table["likely_range_low"] > 0
    return table


# ---------------------------------------------------------------------------
# Baseline rules (same logic as Notebook 04)
# ---------------------------------------------------------------------------
def _share_and_flags(history):
    indexed = history.set_index("week_ending")
    return indexed["icu_occupancy_share_pct"], indexed["is_voluntary_reporting_week"]


def forecast_same_as_this_week(origin_week, target_week, history):
    return history["icu_occupancy_share_pct"].iloc[-1]


def forecast_same_as_last_year(origin_week, target_week, history):
    share_by_week, is_flagged_by_week = _share_and_flags(history)
    for weeks_back in (52, 104):
        earlier_week = target_week - pd.Timedelta(weeks=weeks_back)
        if earlier_week in share_by_week.index and not is_flagged_by_week[earlier_week]:
            return share_by_week[earlier_week]
    return np.nan
