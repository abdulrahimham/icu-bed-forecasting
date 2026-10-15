"""Weekly model inputs, each calculated using only data available at that week."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DATA_FOLDER = Path(__file__).resolve().parent.parent / "data" / "raw"

RESPIRATORY_COLUMNS = {
    "totalconfc19icupats": "covid_icu_patients",
    "totalconffluicupats": "flu_icu_patients",
    "totalconfrsvicupats": "rsv_icu_patients",
    "totalconfc19newadm": "covid_new_admissions",
    "totalconfflunewadm": "flu_new_admissions",
    "totalconfrsvnewadm": "rsv_new_admissions",
}

LEVEL_SEASON_FEATURES = [
    "current_share_pct",
    "share_change_last_1_week",
    "share_change_last_4_weeks",
    "target_season_sin",
    "target_season_cos",
]
RESPIRATORY_FEATURES = [
    "respiratory_icu_share_pct",
    "respiratory_admissions_per_100_icu_beds",
    "admissions_change_last_2_weeks",
]


def load_respiratory_weekly():
    raw_file_path = sorted(RAW_DATA_FOLDER.glob("nhsn_hrd_california_weekly_*.csv"))[-1]
    respiratory = pd.read_csv(raw_file_path, usecols=["weekendingdate", *RESPIRATORY_COLUMNS])
    respiratory = respiratory.rename(columns=RESPIRATORY_COLUMNS)
    respiratory["week_ending"] = pd.to_datetime(respiratory["weekendingdate"]).dt.normalize()
    return respiratory.drop(columns="weekendingdate")


def build_weekly_features(icu_weekly):
    """One row per week with inputs known at that week.

    Flu and RSV were not reported in the earliest weeks (RSV only from late 2023). Those gaps are
    treated as 0, so respiratory counts before late 2023 are a minimum.
    Respiratory counts are divided by ICU beds available so that changes in how many hospitals
    report affect them the same way they affect the share.
    """
    weekly = icu_weekly.merge(load_respiratory_weekly(), on="week_ending", how="left")
    icu_patient_columns = ["covid_icu_patients", "flu_icu_patients", "rsv_icu_patients"]
    admission_columns = ["covid_new_admissions", "flu_new_admissions", "rsv_new_admissions"]
    weekly["respiratory_icu_share_pct"] = (
        weekly[icu_patient_columns].fillna(0).sum(axis=1) / weekly["icu_beds_available"] * 100
    )
    weekly["respiratory_admissions_per_100_icu_beds"] = (
        weekly[admission_columns].fillna(0).sum(axis=1) / weekly["icu_beds_available"] * 100
    )
    weekly["admissions_change_last_2_weeks"] = weekly["respiratory_admissions_per_100_icu_beds"].diff(2)
    weekly["current_share_pct"] = weekly["icu_occupancy_share_pct"]
    weekly["share_change_last_1_week"] = weekly["icu_occupancy_share_pct"].diff(1)
    weekly["share_change_last_4_weeks"] = weekly["icu_occupancy_share_pct"].diff(4)
    # Inputs that reach back into flagged weeks are unreliable
    weekly["inputs_touch_flagged_week"] = (
        weekly["is_voluntary_reporting_week"].astype(int).rolling(5, min_periods=1).max().astype(bool)
    )
    return weekly


def add_target_season(features, target_weeks):
    """Time of year of the target week, as a point on a circle, so late December sits next to early January."""
    day_of_year = pd.DatetimeIndex(target_weeks).dayofyear.to_numpy()
    angle = 2 * np.pi * day_of_year / 365.25
    features = features.copy()
    features["target_season_sin"] = np.sin(angle)
    features["target_season_cos"] = np.cos(angle)
    return features
