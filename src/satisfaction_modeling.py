import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.linalg import qr
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src import model_audit

# -------------------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------------------

warnings.filterwarnings('ignore', category=FutureWarning)

# Shared path to the canonical cleaned respondent-year table
ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / 'data' / 'derived' / 'clean_core.parquet'

# Core random seed and canonical target column names for the satisfaction task
RANDOM_STATE = 42
SAT_TARGET_COL = 'job_sat_std'
SAT_BINARY_COL = 'sat_binary'
SAT_INSTRUMENT_COL = 'job_sat_instrument'

# Canonical year window and held-out split used for future-wave validation
SAT_CANONICAL_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2024, 2025]
SAT_TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2020]
SAT_VALID_YEAR = 2024
SAT_TEST_YEAR = 2025

# Shared 5-class ordinal label setup plus the optional positive/binary cut
SAT_LABELS = [1, 2, 3, 4, 5]
SAT_BINARY_THRESHOLD = 4

# Numeric 0-10 harmonization schemes for the later-wave satisfaction instruments
SAT_NUMERIC_SCHEMES = {
    'default': [1, 4, 6, 8],
    'alt_equal_width': [2, 4, 6, 8]
}

# Survey instrument family used by each year that enters the canonical task
SAT_INSTRUMENTS = {
    2015: 'text_5pt',
    2016: 'text_5pt',
    2017: 'numeric_11pt',
    2018: 'text_7pt',
    2019: 'text_5pt',
    2020: 'text_5pt',
    2024: 'numeric_11pt',
    2025: 'numeric_11pt'
}

# Text label harmonization map that collapses the raw scales into one shared target
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

# Default LightGBM and CatBoost settings for the main satisfaction comparison
DEFAULT_LGB_PARAMS = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': len(SAT_LABELS),
    'n_estimators': 400,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': -1
}
DEFAULT_CATBOOST_PARAMS = {
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 5.0,
    'random_seed': RANDOM_STATE,
    'auto_class_weights': 'Balanced',
    'allow_writing_files': False,
    'verbose': False
}

# OrderedModel gets sampled because the full dummy matrix gets unruly in a hurry
ORDERED_MAX_ROWS = 50_000

# Numeric fields that should always be coerced before modeling or diagnostics
SAT_NUMERIC_FIELDS = [
    'age_mid',
    'years_code_clean',
    'professional_experience_years',
    'years_code_pro_clean',
    'work_exp_clean',
    'career_start_age_est',
    'coding_start_age_est',
    'pro_to_total_code_ratio',
    'language_count',
    'database_count',
    'platform_count',
    'role_family_count',
    'log_comp_real_2025'
]

# Group hierarchy used for the simple categorical mode baseline
SAT_BASELINE_GROUP_SETS = [
    ['country_clean'],
    ['region'],
    []
]


# -------------------------------------------------------------------------------------
# Target build
# -------------------------------------------------------------------------------------

# Maps each survey year to the type of job-satisfaction question it used
def sat_instrument(survey_year):
    return SAT_INSTRUMENTS.get(survey_year, pd.NA)


# Converts numeric 0-10 answers into the shared 1-5 ordinal scale
def numeric_job_sat_to_ordinal(value, scheme='default'):
    if pd.isna(value):
        return np.nan

    cutoffs = SAT_NUMERIC_SCHEMES[scheme]
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan

    if numeric <= cutoffs[0]:
        return 1
    if numeric <= cutoffs[1]:
        return 2
    if numeric <= cutoffs[2]:
        return 3
    if numeric <= cutoffs[3]:
        return 4
    return 5


# Standardizes raw text and numeric answers into one comparable target field
def standardize_job_sat_value(value, survey_year, numeric_scheme='default'):
    if pd.isna(value):
        return np.nan

    instrument = sat_instrument(survey_year)
    if instrument == 'numeric_11pt':
        return numeric_job_sat_to_ordinal(value, scheme=numeric_scheme)

    return SAT_MAP.get(value, np.nan)


# Adds the harmonized satisfaction target while keeping the original survey field untouched
def add_job_sat_std(frame, numeric_scheme='default'):
    out = frame.copy()
    out[SAT_INSTRUMENT_COL] = out['survey_year'].map(sat_instrument)
    out[SAT_TARGET_COL] = [
        standardize_job_sat_value(value, year, numeric_scheme=numeric_scheme)
        for value, year in zip(out['job_sat'], out['survey_year'], strict=False)
    ]
    return out


# Builds the age and career-stage proxies that survived the final feature audit
def add_career_stage_features(frame):
    out = frame.copy()
    age_mid = pd.to_numeric(out.get('age_mid', pd.Series(pd.NA, index=out.index)), errors='coerce')
    years_code = pd.to_numeric(out.get('years_code_clean', pd.Series(pd.NA, index=out.index)), errors='coerce')
    prof_exp = pd.to_numeric(
        out.get('professional_experience_years', pd.Series(pd.NA, index=out.index)),
        errors='coerce'
    )

    out['age_mid'] = age_mid
    out['professional_experience_years'] = prof_exp
    out['career_start_age_est'] = age_mid - prof_exp
    out['coding_start_age_est'] = age_mid - years_code

    ratio = (prof_exp / years_code).where(years_code.ne(0))
    out['pro_to_total_code_ratio'] = pd.Series(ratio, index=out.index).replace([np.inf, -np.inf], np.nan)
    return out


# Main builder for the employed-professional respondent-year modeling frame
def build_satisfaction_frame(
    clean_core,
    include_years=None,
    numeric_scheme='default',
    drop_2018=False,
    require_professional=True,
    require_employed=True,
    add_binary_target=True,
    binary_threshold=SAT_BINARY_THRESHOLD
):
    include_years = SAT_CANONICAL_YEARS if include_years is None else include_years
    frame = add_job_sat_std(clean_core, numeric_scheme=numeric_scheme)
    frame = add_career_stage_features(frame)
    frame = frame.loc[frame['survey_year'].isin(include_years)].copy()
    if drop_2018:
        frame = frame.loc[frame['survey_year'].ne(2018)].copy()
    frame = frame.loc[frame[SAT_TARGET_COL].notna()].copy()

    if require_professional:
        frame = frame.loc[frame['is_professional']].copy()
    if require_employed:
        frame = frame.loc[model_audit.paid_work_mask(frame)].copy()

    frame[SAT_TARGET_COL] = frame[SAT_TARGET_COL].astype(int)
    frame['survey_year_str'] = frame['survey_year'].astype('string')

    if add_binary_target:
        frame[SAT_BINARY_COL] = frame[SAT_TARGET_COL].ge(binary_threshold).astype(int)

    return frame


# Quick year summary for target coverage and population checks
def job_sat_year_summary(
    clean_core,
    include_years=None,
    numeric_scheme='default',
    drop_2018=False
):
    include_years = SAT_CANONICAL_YEARS if include_years is None else include_years
    frame = add_job_sat_std(clean_core, numeric_scheme=numeric_scheme)
    frame = frame.loc[frame['survey_year'].isin(include_years)].copy()
    if drop_2018:
        frame = frame.loc[frame['survey_year'].ne(2018)].copy()

    employed_mask = model_audit.paid_work_mask(frame)
    return (
        frame
        .groupby('survey_year')
        .agg(
            rows=('survey_year', 'size'),
            raw_non_null=('job_sat', lambda series: int(series.notna().sum())),
            harmonized_non_null=(SAT_TARGET_COL, lambda series: int(series.notna().sum())),
            employed_prof_non_null=(
                SAT_TARGET_COL,
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


def job_sat_distribution_by_year(frame, target_col=SAT_TARGET_COL):
    return (
        frame
        .groupby('survey_year')[target_col]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=SAT_LABELS, fill_value=0)
        .reset_index()
    )


def feature_availability_by_year(frame, columns):
    rows = []

    for year in sorted(frame['survey_year'].unique()):
        sub = frame.loc[frame['survey_year'].eq(year)]
        row = {
            'survey_year': year,
            'rows': len(sub)
        }
        for col in columns:
            if col not in sub.columns:
                row[col] = np.nan
            else:
                row[col] = round(float(sub[col].notna().mean()), 3)
        rows.append(row)

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------------
# Feature setup
# -------------------------------------------------------------------------------------
def get_role_cols(frame):
    excluded = {'role_family', 'role_family_count'}
    return sorted(col for col in frame.columns if col.startswith('role_') and col not in excluded)


def build_feature_sets(role_cols):
    core_no_comp_cat = [
        'survey_year_str',
        'region',
        'country_clean',
        'age_group',
        'education_clean',
        'employment_primary',
        'org_size_clean',
        'remote_group'
    ]
    core_no_comp_num = [
        'age_mid',
        'years_code_clean',
        'professional_experience_years',
        'career_start_age_est',
        'coding_start_age_est',
        'pro_to_total_code_ratio',
        'language_count',
        'database_count',
        'platform_count',
        'role_family_count'
    ] + role_cols

    core_with_comp_cat = core_no_comp_cat.copy()
    core_with_comp_num = core_no_comp_num + ['log_comp_real_2025']

    ordered_cat = [
        'survey_year_str',
        'region',
        'age_group',
        'education_clean',
        'employment_primary',
        'org_size_clean',
        'remote_group'
    ]
    ordered_num = ['years_code_clean']

    return {
        'core_no_comp_cat': core_no_comp_cat,
        'core_no_comp_num': core_no_comp_num,
        'core_with_comp_cat': core_with_comp_cat,
        'core_with_comp_num': core_with_comp_num,
        'ordered_cat': ordered_cat,
        'ordered_num': ordered_num
    }


def build_satisfaction_bundle(
    clean_core,
    numeric_scheme='default',
    drop_2018=False
):
    frame = build_satisfaction_frame(clean_core, numeric_scheme=numeric_scheme, drop_2018=drop_2018)
    role_cols = get_role_cols(frame)
    feature_sets = build_feature_sets(role_cols)
    return {
        'frame': frame,
        'role_cols': role_cols,
        'feature_sets': feature_sets
    }


def split_satisfaction_years(
    frame,
    train_years=None,
    valid_year=SAT_VALID_YEAR,
    test_year=SAT_TEST_YEAR
):
    train_years = SAT_TRAIN_YEARS if train_years is None else train_years
    return model_audit.split_years(frame, train_years, valid_year, test_year)


# -------------------------------------------------------------------------------------
# Prep and scoring
# -------------------------------------------------------------------------------------

# Coerces the fields the tree models expect so every fit starts from the same dtypes
def coerce_satisfaction_frame(frame):
    out = frame.copy()

    if 'survey_year_str' not in out.columns:
        out['survey_year_str'] = out['survey_year'].astype('string')

    numeric_fields = SAT_NUMERIC_FIELDS + [col for col in out.columns if col.startswith('role_')]
    for col in numeric_fields:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    if SAT_TARGET_COL in out.columns:
        out[SAT_TARGET_COL] = pd.to_numeric(out[SAT_TARGET_COL], errors='coerce')

    return out


# Saves numeric prep in a way that survives folds where a whole column is missing
def fit_numeric_prep(frame, num_cols):
    observed_cols = [col for col in num_cols if frame[col].notna().any()]
    all_missing_cols = [col for col in num_cols if col not in observed_cols]
    imputer = None

    num_df = pd.DataFrame(index=frame.index)
    if observed_cols:
        imputer = SimpleImputer(strategy='median')
        observed = pd.DataFrame(
            imputer.fit_transform(frame[observed_cols]),
            columns=observed_cols,
            index=frame.index
        )
        num_df[observed_cols] = observed

    for col in all_missing_cols:
        num_df[col] = 0.0

    return num_df.reindex(columns=num_cols), {
        'imputer': imputer,
        'observed_cols': observed_cols,
        'all_missing_cols': all_missing_cols
    }


# Applies the saved numeric prep without losing the all-missing columns
def transform_numeric_prep(frame, num_cols, num_prep):
    observed_cols = num_prep['observed_cols']
    all_missing_cols = num_prep['all_missing_cols']
    imputer = num_prep['imputer']
    num_df = pd.DataFrame(index=frame.index)

    if observed_cols:
        observed = pd.DataFrame(
            imputer.transform(frame[observed_cols]),
            columns=observed_cols,
            index=frame.index
        )
        num_df[observed_cols] = observed

    for col in all_missing_cols:
        num_df[col] = 0.0

    return num_df.reindex(columns=num_cols)


# LightGBM prep path with native categoricals and median-imputed numerics
def prepare_lgbm_frame(frame, cat_cols, num_cols, target_col=SAT_TARGET_COL):
    out = coerce_satisfaction_frame(frame)

    for col in cat_cols:
        out[col] = out[col].astype('string').fillna('Missing').astype('category')

    out_num, num_prep = fit_numeric_prep(out, num_cols)
    out[num_cols] = out_num
    if target_col in out.columns:
        out[target_col] = pd.to_numeric(out[target_col], errors='coerce').astype(int)
    return out, num_prep


# Applies the train-fit LightGBM prep to valid or test rows
def transform_lgbm_frame(frame, cat_cols, num_cols, num_prep, target_col=SAT_TARGET_COL):
    out = coerce_satisfaction_frame(frame)

    for col in cat_cols:
        out[col] = out[col].astype('string').fillna('Missing').astype('category')

    out_num = transform_numeric_prep(out, num_cols, num_prep)
    out[num_cols] = out_num
    if target_col in out.columns:
        out[target_col] = pd.to_numeric(out[target_col], errors='coerce').astype(int)
    return out


# CatBoost gets the same numeric treatment but keeps categoricals as strings
def prepare_catboost_frame(frame, cat_cols, num_cols, target_col=SAT_TARGET_COL, num_prep=None):
    out = coerce_satisfaction_frame(frame)

    for col in cat_cols:
        out[col] = out[col].astype('string').fillna('Missing')

    if num_prep is None:
        out_num, num_prep = fit_numeric_prep(out, num_cols)
    else:
        out_num = transform_numeric_prep(out, num_cols, num_prep)

    out[num_cols] = out_num
    if target_col in out.columns:
        out[target_col] = pd.to_numeric(out[target_col], errors='coerce').astype(int)
    return out, num_prep


def score_satisfaction(y_true, y_pred, labels=SAT_LABELS):
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred, index=y_true.index).astype(int)
    abs_err = (y_true - y_pred).abs()
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'qwk': cohen_kappa_score(y_true, y_pred, labels=labels, weights='quadratic'),
        'adjacent_error_rate': float(abs_err.eq(1).mean()),
        'far_miss_rate': float(abs_err.ge(2).mean())
    }

    for label, recall in zip(labels, recalls, strict=False):
        metrics[f'recall_{label}'] = float(recall)

    return metrics


def confusion_frame(y_true, y_pred, labels=SAT_LABELS):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def subgroup_metrics(frame, y_true, y_pred, group_cols):
    data = frame.loc[:, group_cols].copy()
    data['y_true'] = pd.Series(y_true, index=frame.index).astype(int)
    data['y_pred'] = pd.Series(y_pred, index=frame.index).astype(int)
    rows = []

    for key, sub in data.groupby(group_cols, dropna=False):
        if len(group_cols) == 1:
            key = [key[0] if isinstance(key, tuple) else key]
        else:
            key = list(key)
        row = dict(zip(group_cols, key, strict=False))
        row['rows'] = len(sub)
        row.update(score_satisfaction(sub['y_true'], sub['y_pred']))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    sort_cols = []
    for col in group_cols:
        sort_col = f'__sort_{col}'
        out[sort_col] = out[col].astype('string').fillna('Missing')
        sort_cols.append(sort_col)

    return out.sort_values(sort_cols).drop(columns=sort_cols).reset_index(drop=True)


# -------------------------------------------------------------------------------------
# Baselines
# -------------------------------------------------------------------------------------
def predict_global_mode(train_df, score_df, target_col=SAT_TARGET_COL):
    mode = int(train_df[target_col].mode().sort_values().iloc[0])
    return pd.Series(mode, index=score_df.index, dtype='int64')


def predict_hier_mode(train_df, score_df, group_sets=None, target_col=SAT_TARGET_COL):
    group_sets = SAT_BASELINE_GROUP_SETS if group_sets is None else group_sets
    pred = pd.Series(np.nan, index=score_df.index, dtype='float64')

    for group_cols in group_sets:
        if not group_cols:
            fallback = int(train_df[target_col].mode().sort_values().iloc[0])
            pred = pred.fillna(fallback)
            continue

        modes = (
            train_df
            .groupby(group_cols, dropna=False)[target_col]
            .agg(lambda series: int(series.mode().sort_values().iloc[0]))
            .rename('pred')
            .reset_index()
        )
        scored = score_df[group_cols].merge(modes, on=group_cols, how='left')
        pred = pred.fillna(pd.Series(scored['pred'].to_numpy(), index=score_df.index))

    return pred.astype(int)


def score_mode_baselines(train_df, valid_df, test_df, group_sets=None, target_col=SAT_TARGET_COL):
    global_valid = predict_global_mode(train_df, valid_df, target_col=target_col)
    global_test = predict_global_mode(pd.concat([train_df, valid_df], axis=0), test_df, target_col=target_col)
    hier_valid = predict_hier_mode(train_df, valid_df, group_sets=group_sets, target_col=target_col)
    hier_test = predict_hier_mode(pd.concat([train_df, valid_df], axis=0), test_df, group_sets=group_sets, target_col=target_col)

    return {
        'global_mode': {
            'valid_metrics': score_satisfaction(valid_df[target_col], global_valid),
            'test_metrics': score_satisfaction(test_df[target_col], global_test),
            'valid_pred': global_valid,
            'test_pred': global_test
        },
        'hier_mode': {
            'valid_metrics': score_satisfaction(valid_df[target_col], hier_valid),
            'test_metrics': score_satisfaction(test_df[target_col], hier_test),
            'valid_pred': hier_valid,
            'test_pred': hier_test
        }
    }


# -------------------------------------------------------------------------------------
# Ordered baseline
# -------------------------------------------------------------------------------------
def sample_ordered_frame(frame, target_col=SAT_TARGET_COL, max_rows=ORDERED_MAX_ROWS):
    if len(frame) <= max_rows:
        return frame.copy()

    strata = frame[['survey_year', target_col]].astype('string').agg('|'.join, axis=1)
    sampled = (
        frame
        .assign(_strata=strata)
        .groupby('_strata', group_keys=False)
        .apply(
            lambda sub: sub.sample(
                n=max(1, int(round(max_rows * len(sub) / len(frame)))),
                random_state=RANDOM_STATE
            )
        )
    )

    if isinstance(sampled.index, pd.MultiIndex):
        sampled = sampled.reset_index(level=0, drop=True)
    if '_strata' in sampled.columns:
        sampled = sampled.drop(columns='_strata')

    if len(sampled) > max_rows:
        sampled = sampled.sample(max_rows, random_state=RANDOM_STATE)

    return sampled.sort_index()


def ordered_design_matrix(frame, cat_cols, num_cols, fit_columns=None, fill_values=None):
    out = frame[cat_cols + num_cols].copy()

    for col in cat_cols:
        out[col] = out[col].astype('string').fillna('Missing')

    fill_values = {} if fill_values is None else dict(fill_values)
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors='coerce')
        if col not in fill_values:
            fill_values[col] = float(out[col].median()) if out[col].notna().any() else 0.0
        out[col] = out[col].fillna(fill_values[col])

    design = pd.get_dummies(out, columns=cat_cols, drop_first=True, dtype=float)
    if fit_columns is None:
        fit_columns = [col for col in design.columns if design[col].nunique(dropna=False) > 1]
        if fit_columns:
            design_fit = design[fit_columns]
            rank = np.linalg.matrix_rank(design_fit.to_numpy(dtype=float))
            if rank < len(fit_columns):
                _, _, pivots = qr(design_fit.to_numpy(dtype=float), mode='economic', pivoting=True)
                keep_idx = sorted(pivots[:rank])
                fit_columns = [fit_columns[idx] for idx in keep_idx]
    design = design.reindex(columns=fit_columns, fill_value=0.0)
    return design.astype(float), fit_columns, fill_values


def ordered_predict(result, design):
    probs = result.model.predict(result.params, exog=design)
    labels = np.asarray(SAT_LABELS)
    return pd.Series(labels[np.asarray(probs).argmax(axis=1)], index=design.index)


def fit_ordered_baseline(
    train_df,
    valid_df,
    test_df,
    cat_cols,
    num_cols,
    target_col=SAT_TARGET_COL,
    maxiter=200,
    max_train_rows=ORDERED_MAX_ROWS
):
    fallback_specs = [
        {
            'cat_cols': list(cat_cols),
            'num_cols': list(num_cols)
        },
        {
            'cat_cols': [col for col in cat_cols if col in {'survey_year_str', 'region', 'education_clean'}],
            'num_cols': [col for col in num_cols if col in {'years_code_clean', 'role_family_count'}]
        },
        {
            'cat_cols': [col for col in cat_cols if col in {'survey_year_str', 'region'}],
            'num_cols': [col for col in num_cols if col in {'years_code_clean'}]
        }
    ]

    last_error = None

    for level, spec in enumerate(fallback_specs):
        if not spec['cat_cols'] and not spec['num_cols']:
            continue

        try:
            train_fit = sample_ordered_frame(train_df, target_col=target_col, max_rows=max_train_rows)
            train_design, fit_columns, fill_values = ordered_design_matrix(
                train_fit,
                spec['cat_cols'],
                spec['num_cols']
            )
            valid_design, _, _ = ordered_design_matrix(
                valid_df,
                spec['cat_cols'],
                spec['num_cols'],
                fit_columns=fit_columns,
                fill_values=fill_values
            )

            train_y = train_fit[target_col].astype(int)
            valid_y = valid_df[target_col].astype(int)
            model = OrderedModel(train_y, train_design, distr='logit')
            result = model.fit(method='lbfgs', maxiter=maxiter, disp=False)
            valid_pred = ordered_predict(result, valid_design)
            valid_metrics = score_satisfaction(valid_y, valid_pred)

            train_valid = pd.concat([train_df, valid_df], axis=0)
            train_valid_fit = sample_ordered_frame(train_valid, target_col=target_col, max_rows=max_train_rows)
            final_design, final_columns, final_fill = ordered_design_matrix(
                train_valid_fit,
                spec['cat_cols'],
                spec['num_cols']
            )
            test_design, _, _ = ordered_design_matrix(
                test_df,
                spec['cat_cols'],
                spec['num_cols'],
                fit_columns=final_columns,
                fill_values=final_fill
            )
            final_model = OrderedModel(train_valid_fit[target_col].astype(int), final_design, distr='logit')
            final_result = final_model.fit(method='lbfgs', maxiter=maxiter, disp=False)
            test_pred = ordered_predict(final_result, test_design)
            test_metrics = score_satisfaction(test_df[target_col], test_pred)

            return {
                'model': final_result,
                'valid_metrics': valid_metrics,
                'test_metrics': test_metrics,
                'valid_pred': valid_pred,
                'test_pred': test_pred,
                'feature_cols': fit_columns,
                'fit_rows': len(train_fit),
                'final_fit_rows': len(train_valid_fit),
                'ordered_cat_cols': spec['cat_cols'],
                'ordered_num_cols': spec['num_cols'],
                'fallback_level': level
            }
        except ValueError as exc:
            last_error = exc
            continue

    raise last_error


# -------------------------------------------------------------------------------------
# LightGBM
# -------------------------------------------------------------------------------------
def resolve_lgb_params(params=None):
    resolved = dict(DEFAULT_LGB_PARAMS)
    if params is not None:
        resolved.update(params)
    return resolved


def tune_lgbm_multiclass(
    train_df,
    valid_df,
    cat_cols,
    num_cols,
    base_params=None,
    n_trials=20
):
    resolved = resolve_lgb_params(base_params)
    train_prepped, num_imputer = prepare_lgbm_frame(train_df, cat_cols, num_cols)
    valid_prepped = transform_lgbm_frame(valid_df, cat_cols, num_cols, num_imputer)
    features = cat_cols + num_cols

    def objective(trial):
        params = dict(resolved)
        params.update({
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
        })
        model = lgb.LGBMClassifier(**params)
        model.fit(
            train_prepped[features],
            train_prepped[SAT_TARGET_COL].to_numpy(),
            eval_set=[(valid_prepped[features], valid_prepped[SAT_TARGET_COL].to_numpy())],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        pred = model.predict(valid_prepped[features], num_iteration=model.best_iteration_)
        return score_satisfaction(valid_prepped[SAT_TARGET_COL], pred)['qwk']

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best = dict(resolved)
    best.update(study.best_trial.params)
    return best


def fit_lgbm_multiclass(
    train_df,
    valid_df,
    test_df,
    cat_cols,
    num_cols,
    params=None,
    tune_trials=0,
    early_stopping_rounds=50
):
    resolved = resolve_lgb_params(params)
    if tune_trials:
        resolved = tune_lgbm_multiclass(train_df, valid_df, cat_cols, num_cols, base_params=resolved, n_trials=tune_trials)

    train_prepped, num_imputer = prepare_lgbm_frame(train_df, cat_cols, num_cols)
    valid_prepped = transform_lgbm_frame(valid_df, cat_cols, num_cols, num_imputer)
    features = cat_cols + num_cols

    early_model = lgb.LGBMClassifier(**resolved)
    early_model.fit(
        train_prepped[features],
        train_prepped[SAT_TARGET_COL].to_numpy(),
        eval_set=[(valid_prepped[features], valid_prepped[SAT_TARGET_COL].to_numpy())],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
    )
    best_iteration = early_model.best_iteration_ or resolved['n_estimators']
    valid_pred = pd.Series(
        early_model.predict(valid_prepped[features], num_iteration=best_iteration),
        index=valid_prepped.index
    ).astype(int)
    valid_metrics = score_satisfaction(valid_prepped[SAT_TARGET_COL], valid_pred)

    train_valid = pd.concat([train_df, valid_df], axis=0)
    final_params = dict(resolved)
    final_params['n_estimators'] = best_iteration
    final_prepped, final_imputer = prepare_lgbm_frame(train_valid, cat_cols, num_cols)
    test_prepped = transform_lgbm_frame(test_df, cat_cols, num_cols, final_imputer)
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(final_prepped[features], final_prepped[SAT_TARGET_COL].to_numpy())
    test_pred = pd.Series(final_model.predict(test_prepped[features]), index=test_prepped.index).astype(int)
    test_metrics = score_satisfaction(test_prepped[SAT_TARGET_COL], test_pred)

    return {
        'model': final_model,
        'valid_metrics': valid_metrics,
        'test_metrics': test_metrics,
        'valid_pred': valid_pred,
        'test_pred': test_pred,
        'feature_cols': features,
        'best_params': final_params
    }


# -------------------------------------------------------------------------------------
# CatBoost
# -------------------------------------------------------------------------------------
def resolve_catboost_params(params=None):
    resolved = dict(DEFAULT_CATBOOST_PARAMS)
    if params is not None:
        resolved.update(params)
    return resolved


def fit_catboost_multiclass(
    train_df,
    valid_df,
    test_df,
    cat_cols,
    num_cols,
    params=None,
    early_stopping_rounds=50
):
    resolved = resolve_catboost_params(params)
    train_prepped, num_prep = prepare_catboost_frame(train_df, cat_cols, num_cols)
    valid_prepped, _ = prepare_catboost_frame(valid_df, cat_cols, num_cols, num_prep=num_prep)
    features = cat_cols + num_cols
    train_pool = Pool(train_prepped[features], train_prepped[SAT_TARGET_COL], cat_features=cat_cols)
    valid_pool = Pool(valid_prepped[features], valid_prepped[SAT_TARGET_COL], cat_features=cat_cols)

    early_model = CatBoostClassifier(**resolved)
    early_model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=early_stopping_rounds)
    valid_pred = pd.Series(early_model.predict(valid_pool).reshape(-1), index=valid_prepped.index).astype(int)
    valid_metrics = score_satisfaction(valid_prepped[SAT_TARGET_COL], valid_pred)

    best_iteration = early_model.get_best_iteration()
    if best_iteration is None or best_iteration <= 0:
        best_iteration = resolved['iterations']

    train_valid = pd.concat([train_df, valid_df], axis=0)
    final_prepped, final_num_prep = prepare_catboost_frame(train_valid, cat_cols, num_cols)
    test_prepped, _ = prepare_catboost_frame(test_df, cat_cols, num_cols, num_prep=final_num_prep)
    final_pool = Pool(final_prepped[features], final_prepped[SAT_TARGET_COL], cat_features=cat_cols)
    test_pool = Pool(test_prepped[features], test_prepped[SAT_TARGET_COL], cat_features=cat_cols)

    final_params = dict(resolved)
    final_params['iterations'] = best_iteration
    final_model = CatBoostClassifier(**final_params)
    final_model.fit(final_pool, verbose=False)
    test_pred = pd.Series(final_model.predict(test_pool).reshape(-1), index=test_prepped.index).astype(int)
    test_metrics = score_satisfaction(test_prepped[SAT_TARGET_COL], test_pred)

    return {
        'model': final_model,
        'valid_metrics': valid_metrics,
        'test_metrics': test_metrics,
        'valid_pred': valid_pred,
        'test_pred': test_pred,
        'feature_cols': features,
        'best_params': final_params
    }


# -------------------------------------------------------------------------------------
# Comparison and sensitivities
# -------------------------------------------------------------------------------------

# Flattens the model outputs into the table format used in notebooks and reports
def result_row(setup, result):
    if result is None:
        return {
            'setup': setup,
            'valid_qwk': np.nan,
            'valid_macro_f1': np.nan,
            'test_qwk': np.nan,
            'test_macro_f1': np.nan,
            'test_weighted_f1': np.nan,
            'test_accuracy': np.nan,
            'test_adjacent_error_rate': np.nan,
            'test_far_miss_rate': np.nan
        }

    return {
        'setup': setup,
        'valid_qwk': result['valid_metrics']['qwk'],
        'valid_macro_f1': result['valid_metrics']['macro_f1'],
        'test_qwk': result['test_metrics']['qwk'],
        'test_macro_f1': result['test_metrics']['macro_f1'],
        'test_weighted_f1': result['test_metrics']['weighted_f1'],
        'test_accuracy': result['test_metrics']['accuracy'],
        'test_adjacent_error_rate': result['test_metrics']['adjacent_error_rate'],
        'test_far_miss_rate': result['test_metrics']['far_miss_rate']
    }


# Runs the canonical satisfaction comparison table plus the with-comp sensitivity branch
def compare_satisfaction_setups(
    clean_core,
    numeric_scheme='default',
    drop_2018=False,
    lgb_params=None,
    catboost_params=None,
    tune_lgb_trials=0
):
    bundle = build_satisfaction_bundle(clean_core, numeric_scheme=numeric_scheme, drop_2018=drop_2018)
    feature_sets = bundle['feature_sets']
    frame = bundle['frame']
    train_df, valid_df, test_df = split_satisfaction_years(frame)
    baselines = score_mode_baselines(train_df, valid_df, test_df)
    ordered = fit_ordered_baseline(
        train_df,
        valid_df,
        test_df,
        feature_sets['ordered_cat'],
        feature_sets['ordered_num']
    )
    lgb_core = fit_lgbm_multiclass(
        train_df,
        valid_df,
        test_df,
        feature_sets['core_no_comp_cat'],
        feature_sets['core_no_comp_num'],
        params=lgb_params,
        tune_trials=tune_lgb_trials
    )
    catboost_core = fit_catboost_multiclass(
        train_df,
        valid_df,
        test_df,
        feature_sets['core_no_comp_cat'],
        feature_sets['core_no_comp_num'],
        params=catboost_params
    )
    selected_main_setup = pd.DataFrame([
        result_row('LightGBM core_no_comp', lgb_core),
        result_row('CatBoost core_no_comp', catboost_core)
    ]).sort_values(['valid_qwk', 'valid_macro_f1'], ascending=False)['setup'].iloc[0]
    selected_main_result = {
        'LightGBM core_no_comp': lgb_core,
        'CatBoost core_no_comp': catboost_core
    }[selected_main_setup]
    selected_family = 'catboost' if selected_main_setup.startswith('CatBoost') else 'lightgbm'
    selected_family_name = selected_main_setup.replace(' core_no_comp', '')
    comp_subset_setup_name = f'{selected_family_name} core_no_comp (comp subset)'
    with_comp_setup_name = f'{selected_family_name} core_with_comp'

    comp_frame = frame.loc[frame['log_comp_real_2025'].notna()].copy()
    selected_family_comp_subset_no_comp = None
    selected_family_with_comp = None
    with_comp_counts = pd.DataFrame(columns=['survey_year', 'rows'])
    if len(comp_frame):
        comp_train, comp_valid, comp_test = split_satisfaction_years(comp_frame)
        if len(comp_train) and len(comp_valid) and len(comp_test):
            if selected_family == 'catboost':
                selected_family_comp_subset_no_comp = fit_catboost_multiclass(
                    comp_train,
                    comp_valid,
                    comp_test,
                    feature_sets['core_no_comp_cat'],
                    feature_sets['core_no_comp_num'],
                    params=catboost_params
                )
                selected_family_with_comp = fit_catboost_multiclass(
                    comp_train,
                    comp_valid,
                    comp_test,
                    feature_sets['core_with_comp_cat'],
                    feature_sets['core_with_comp_num'],
                    params=catboost_params
                )
            else:
                selected_family_comp_subset_no_comp = fit_lgbm_multiclass(
                    comp_train,
                    comp_valid,
                    comp_test,
                    feature_sets['core_no_comp_cat'],
                    feature_sets['core_no_comp_num'],
                    params=lgb_params,
                    tune_trials=0
                )
                selected_family_with_comp = fit_lgbm_multiclass(
                    comp_train,
                    comp_valid,
                    comp_test,
                    feature_sets['core_with_comp_cat'],
                    feature_sets['core_with_comp_num'],
                    params=lgb_params,
                    tune_trials=0
                )
        with_comp_counts = comp_frame.groupby('survey_year').size().rename('rows').reset_index()

    summary = pd.DataFrame([
        result_row('Global mode baseline', baselines['global_mode']),
        result_row('Country-region mode baseline', baselines['hier_mode']),
        result_row('Ordered baseline', ordered),
        result_row('LightGBM core_no_comp', lgb_core),
        result_row('CatBoost core_no_comp', catboost_core),
        result_row(comp_subset_setup_name, selected_family_comp_subset_no_comp),
        result_row(with_comp_setup_name, selected_family_with_comp)
    ]).sort_values(['valid_qwk', 'valid_macro_f1'], ascending=False)

    return {
        'bundle': bundle,
        'summary': summary,
        'baselines': baselines,
        'ordered': ordered,
        'lgb_core': lgb_core,
        'catboost_core': catboost_core,
        'selected_family': selected_family,
        'selected_family_name': selected_family_name,
        'selected_family_comp_subset_setup': comp_subset_setup_name,
        'selected_family_with_comp_setup': with_comp_setup_name,
        'selected_family_comp_subset_no_comp': selected_family_comp_subset_no_comp,
        'selected_family_with_comp': selected_family_with_comp,
        'lgb_comp_subset_no_comp': selected_family_comp_subset_no_comp if selected_family == 'lightgbm' else None,
        'lgb_with_comp': selected_family_with_comp if selected_family == 'lightgbm' else None,
        'frame_counts': frame.groupby('survey_year').size().rename('rows').reset_index(),
        'with_comp_counts': with_comp_counts,
        'selected_main_setup': selected_main_setup,
        'selected_main_result': selected_main_result
    }


# Rolling-origin helper so we can sanity check whether the chosen family stays stable over time
def rolling_origin_satisfaction(
    clean_core,
    spec='core_no_comp',
    model_family='lightgbm',
    numeric_scheme='default',
    drop_2018=False,
    lgb_params=None,
    catboost_params=None,
    min_train_year=2015,
    final_valid_year=2024
):
    bundle = build_satisfaction_bundle(clean_core, numeric_scheme=numeric_scheme, drop_2018=drop_2018)
    frame = bundle['frame']
    feature_sets = bundle['feature_sets']
    cat_cols = feature_sets[f'{spec}_cat'] if spec != 'ordered' else feature_sets['ordered_cat']
    num_cols = feature_sets[f'{spec}_num'] if spec != 'ordered' else feature_sets['ordered_num']

    if spec == 'core_with_comp':
        frame = frame.loc[frame['log_comp_real_2025'].notna()].copy()

    rows = []
    for train_years, valid_year, train_df, valid_df in model_audit.rolling_origin_splits(
        frame,
        min_train_year=min_train_year,
        final_valid_year=final_valid_year
    ):
        if model_family == 'ordered':
            result = fit_ordered_baseline(train_df, valid_df, valid_df, cat_cols, num_cols)
        elif model_family == 'catboost':
            result = fit_catboost_multiclass(train_df, valid_df, valid_df, cat_cols, num_cols, params=catboost_params)
        else:
            result = fit_lgbm_multiclass(train_df, valid_df, valid_df, cat_cols, num_cols, params=lgb_params)

        rows.append({
            'train_years': ','.join(map(str, train_years)),
            'valid_year': valid_year,
            **result['valid_metrics']
        })

    return pd.DataFrame(rows)


# Checks whether alternate target harmonization choices materially change the modeling story
def harmonization_sensitivity(
    clean_core,
    spec='core_no_comp',
    model_family='lightgbm',
    lgb_params=None,
    catboost_params=None,
    tune_lgb_trials=0
):
    setup_name = {
        'lightgbm': f'LightGBM {spec}',
        'catboost': f'CatBoost {spec}',
        'ordered': 'Ordered baseline'
    }[model_family]
    rows = []

    for numeric_scheme, drop_2018 in [
        ('default', False),
        ('alt_equal_width', False),
        ('default', True)
    ]:
        bundle = build_satisfaction_bundle(clean_core, numeric_scheme=numeric_scheme, drop_2018=drop_2018)
        frame = bundle['frame']
        feature_sets = bundle['feature_sets']

        if spec == 'core_with_comp':
            frame = frame.loc[frame['log_comp_real_2025'].notna()].copy()

        train_df, valid_df, test_df = split_satisfaction_years(frame)
        if model_family == 'ordered':
            result = fit_ordered_baseline(
                train_df,
                valid_df,
                test_df,
                feature_sets['ordered_cat'],
                feature_sets['ordered_num']
            )
        elif model_family == 'catboost':
            result = fit_catboost_multiclass(
                train_df,
                valid_df,
                test_df,
                feature_sets[f'{spec}_cat'],
                feature_sets[f'{spec}_num'],
                params=catboost_params
            )
        else:
            result = fit_lgbm_multiclass(
                train_df,
                valid_df,
                test_df,
                feature_sets[f'{spec}_cat'],
                feature_sets[f'{spec}_num'],
                params=lgb_params,
                tune_trials=tune_lgb_trials
            )

        rows.append({
            'numeric_scheme': numeric_scheme,
            'drop_2018': drop_2018,
            'setup': setup_name,
            'train_rows': len(train_df),
            'valid_rows': len(valid_df),
            'test_rows': len(test_df),
            'valid_qwk': result['valid_metrics']['qwk'],
            'test_qwk': result['test_metrics']['qwk'],
            'test_macro_f1': result['test_metrics']['macro_f1']
        })

    return pd.DataFrame(rows).sort_values(['valid_qwk', 'test_qwk'], ascending=False)


# -------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------

# Runs the canonical satisfaction comparison directly from the cleaned parquet
def main():
    from src import comp_clean

    clean_core = comp_clean.load_clean_core(CLEAN_PATH)
    results = compare_satisfaction_setups(clean_core)

    print(results['summary'].to_string(index=False))
    print(f"Selected main setup: {results['selected_main_setup']}")


if __name__ == '__main__':
    main()
