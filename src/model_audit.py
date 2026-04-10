from pathlib import Path

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

# -------------------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------------------

# Shared path to the canonical cleaned respondent-year table
ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / 'data' / 'derived' / 'clean_core.parquet'

# Canonical compensation target names reused across the modeling modules
COMP_TARGET_LOG = 'log_comp_real_2025'
COMP_TARGET_REAL = 'comp_real_2025'

# Stable ordinal label set for the harmonized job-satisfaction target
SAT_LABELS = [1, 2, 3, 4, 5]


# -------------------------------------------------------------------------------------
# Shared split helpers
# -------------------------------------------------------------------------------------

# Quick paid-work check so the notebooks and models stay aligned on who counts as employed
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


# Simple year-aware split helper used across both target families
def split_years(frame, train_years, valid_year, test_year):
    train = frame[frame['survey_year'].isin(train_years)].copy()
    valid = frame[frame['survey_year'] == valid_year].copy()
    test = frame[frame['survey_year'] == test_year].copy()
    return train, valid, test


# Rolling-origin generator so future-wave validation stays honest
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


# -------------------------------------------------------------------------------------
# Compensation metrics
# -------------------------------------------------------------------------------------

# Converts log-comp predictions back to real dollars so the metrics read like pay, not math
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


# Hierarchical medians give us the simple geography benchmark for compensation
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


# Wraps the geography median predictor in the same metric bundle as the model code
def score_hier_median(train_df, score_df, group_sets, target_col=COMP_TARGET_LOG):
    pred_log = predict_hier_median(train_df, score_df, group_sets)
    return score_log_comp(score_df[target_col], pred_log)


# -------------------------------------------------------------------------------------
# Satisfaction wrappers
# -------------------------------------------------------------------------------------

# Thin wrapper so older modules can still call the canonical satisfaction harmonizer here
def standardize_job_sat_value(value, survey_year):
    from src import satisfaction_modeling

    return satisfaction_modeling.standardize_job_sat_value(value, survey_year)


# Adds the harmonized job-satisfaction target without forcing callers to import the whole module
def add_job_sat_std(frame):
    from src import satisfaction_modeling

    return satisfaction_modeling.add_job_sat_std(frame)


# Builds the employed-professional satisfaction frame from the cleaned respondent-year table
def build_job_sat_model_frame(
    clean_core,
    include_years=None,
    require_professional=True,
    require_employed=True
):
    from src import satisfaction_modeling

    return satisfaction_modeling.build_satisfaction_frame(
        clean_core,
        include_years=include_years,
        require_professional=require_professional,
        require_employed=require_employed
    )


# Shared year summary wrapper for quick satisfaction audits
def job_sat_year_summary(clean_core):
    from src import satisfaction_modeling

    return satisfaction_modeling.job_sat_year_summary(clean_core)


# Keeps the main ordinal metrics in one compact bundle for diagnostics and tables
def score_sat_classification(y_true, y_pred, labels=SAT_LABELS):
    from src import satisfaction_modeling

    scored = satisfaction_modeling.score_satisfaction(y_true, y_pred, labels=labels)
    return {
        'accuracy': scored['accuracy'],
        'macro_f1': scored['macro_f1'],
        'weighted_f1': scored['weighted_f1'],
        'qwk': scored['qwk']
    }


# -------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------

# Lightweight smoke run so the shared audit helpers can be executed directly when needed
def main():
    from src import comp_clean

    clean_core = comp_clean.load_clean_core(CLEAN_PATH)
    sat_frame = build_job_sat_model_frame(clean_core)

    summary = pd.DataFrame([
        {
            'artifact': 'clean_core',
            'rows': len(clean_core),
            'survey_years': clean_core['survey_year'].nunique()
        },
        {
            'artifact': 'comp_model_sample',
            'rows': int(clean_core['is_comp_model_sample'].fillna(False).sum()),
            'survey_years': clean_core.loc[clean_core['is_comp_model_sample'].fillna(False), 'survey_year'].nunique()
        },
        {
            'artifact': 'job_sat_model_frame',
            'rows': len(sat_frame),
            'survey_years': sat_frame['survey_year'].nunique()
        }
    ])

    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
