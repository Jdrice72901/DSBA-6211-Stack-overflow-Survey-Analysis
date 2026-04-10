import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

COMP_TARGET_LOG = 'log_comp_real_2025'
COMP_TARGET_REAL = 'comp_real_2025'
SAT_LABELS = [1, 2, 3, 4, 5]

SAT_MAP = {
    "I hate my job": 1,
    "I'm somewhat dissatisfied with my job": 2,
    "I'm neither satisfied nor dissatisfied with my job": 3,
    "I'm neither satisfied nor dissatisfied": 3,
    "I'm somewhat satisfied with my job": 4,
    "I love my job": 5,
    "I don't have a job": np.nan,
    "Other (please specify)": np.nan,
    'Extremely dissatisfied': 1,
    'Moderately dissatisfied': 2,
    'Slightly dissatisfied': 2,
    'Neither satisfied nor dissatisfied': 3,
    'Slightly satisfied': 4,
    'Moderately satisfied': 4,
    'Extremely satisfied': 5,
    'Very dissatisfied': 1,
    'Very satisfied': 5
}

JOB_SAT_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2024, 2025]


def paid_work_mask(frame):
    flag_cols = [
        'is_full_time_employed',
        'is_part_time_employed',
        'is_independent'
    ]
    if set(flag_cols).issubset(frame.columns):
        return frame[flag_cols].fillna(False).astype(bool).any(axis=1)

    return frame['employment_group'].isin([
        'Employed full-time',
        'Employed part-time',
        'Independent / contract'
    ])


def split_years(frame, train_years, valid_year, test_year):
    train = frame[frame['survey_year'].isin(train_years)].copy()
    valid = frame[frame['survey_year'] == valid_year].copy()
    test = frame[frame['survey_year'] == test_year].copy()
    return train, valid, test


def rolling_origin_splits(frame, min_train_year=2019, final_valid_year=2024):
    years = sorted(year for year in frame['survey_year'].dropna().unique() if year >= min_train_year)
    for valid_year in years:
        if valid_year > final_valid_year:
            continue
        train_years = [year for year in years if year < valid_year]
        if not train_years:
            continue
        train = frame[frame['survey_year'].isin(train_years)].copy()
        valid = frame[frame['survey_year'] == valid_year].copy()
        if len(train) and len(valid):
            yield train_years, valid_year, train, valid


def score_log_comp(y_true_log, pred_log):
    y_true_real = np.exp(y_true_log)
    pred_real = np.exp(pred_log)
    return {
        'medae_real': median_absolute_error(y_true_real, pred_real),
        'mae_real': mean_absolute_error(y_true_real, pred_real),
        'rmse_real': np.sqrt(mean_squared_error(y_true_real, pred_real)),
        'rmse_log': np.sqrt(mean_squared_error(y_true_log, pred_log)),
        'r2_log': r2_score(y_true_log, pred_log)
    }


def predict_hier_median(train_df, score_df, group_sets, target_col=COMP_TARGET_REAL):
    pred = pd.Series(np.nan, index=score_df.index, dtype='float64')

    for group_cols in group_sets:
        if not group_cols:
            pred = pred.fillna(train_df[target_col].median())
            continue

        medians = (
            train_df
            .groupby(group_cols, dropna=False)[target_col]
            .median()
            .rename('pred_real')
            .reset_index()
        )
        scored = score_df[group_cols].merge(medians, on=group_cols, how='left')
        pred = pred.fillna(pd.Series(scored['pred_real'].to_numpy(), index=score_df.index))

    pred = pred.fillna(train_df[target_col].median())
    return np.log(pred)


def score_hier_median(train_df, score_df, group_sets, target_col=COMP_TARGET_LOG):
    pred_log = predict_hier_median(train_df, score_df, group_sets)
    return score_log_comp(score_df[target_col], pred_log)


def standardize_job_sat_value(value, survey_year):
    if pd.isna(value):
        return np.nan

    if survey_year in {2017, 2024, 2025}:
        try:
            numeric = float(value)
        except ValueError:
            return np.nan
        if numeric <= 1:
            return 1
        if numeric <= 4:
            return 2
        if numeric <= 6:
            return 3
        if numeric <= 8:
            return 4
        return 5

    return SAT_MAP.get(value, np.nan)


def add_job_sat_std(frame):
    out = frame.copy()
    out['job_sat_std'] = [
        standardize_job_sat_value(value, year)
        for value, year in zip(out['job_sat'], out['survey_year'], strict=False)
    ]
    return out


def build_job_sat_model_frame(
    clean_core,
    include_years=None,
    require_professional=True,
    require_employed=True
):
    include_years = JOB_SAT_YEARS if include_years is None else include_years
    frame = add_job_sat_std(clean_core)
    frame = frame.loc[frame['survey_year'].isin(include_years)].copy()
    frame = frame.loc[frame['job_sat_std'].notna()].copy()

    if require_professional:
        frame = frame.loc[frame['is_professional']].copy()
    if require_employed:
        frame = frame.loc[paid_work_mask(frame)].copy()

    return frame


def job_sat_year_summary(clean_core):
    frame = add_job_sat_std(clean_core)
    employed_mask = paid_work_mask(frame)
    return (
        frame
        .groupby('survey_year')
        .agg(
            rows=('survey_year', 'size'),
            raw_non_null=('job_sat', lambda series: int(series.notna().sum())),
            harmonized_non_null=('job_sat_std', lambda series: int(series.notna().sum())),
            employed_prof_non_null=(
                'job_sat_std',
                lambda series: int(
                    series.loc[
                        frame.loc[series.index, 'is_professional']
                        & employed_mask.loc[series.index]
                    ].notna().sum()
                )
            )
        )
        .reset_index()
    )


def score_sat_classification(y_true, y_pred, labels=SAT_LABELS):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'qwk': cohen_kappa_score(y_true, y_pred, labels=labels, weights='quadratic')
    }
