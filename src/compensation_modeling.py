import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from matplotlib.ticker import FuncFormatter
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import comp_clean, model_audit

# -------------------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------------------

# Shared path to the canonical cleaned respondent year table
ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / 'data' / 'derived' / 'clean_core.parquet'
OUTPUT_ROOT = ROOT / 'data' / 'outputs'
REPORT_DIR = OUTPUT_ROOT / 'compensation_reporting'
REPORT_US_DIR = OUTPUT_ROOT / 'compensation_reporting_us'

# Core random seed and target columns used throughout the compensation workflow
RANDOM_STATE = 42
TARGET_COL = model_audit.COMP_TARGET_LOG
REAL_TARGET_COL = model_audit.COMP_TARGET_REAL
REAL_WINSOR_COL = 'comp_real_2025_winsor'
WINSOR_TARGET_COL = 'log_comp_real_2025_winsor'

# The year windows that define the comparable core, tech rich, and AI era runs
CORE_WINDOW_YEARS = [2019, 2020, 2021, 2022, 2023]
TECH_WINDOW_YEARS = [2021, 2022, 2023]
AI_WINDOW_YEARS = [2023]
VALID_YEAR = 2024
TEST_YEAR = 2025

# How many top tech tokens to keep when turning the multiselects into boolean flags
TOP_N_TECH = {
    'language': 15,
    'database': 10,
    'platform': 10
}

# Default reporting controls for the locked compensation model visuals
REPORT_SHAP_SAMPLE = 2000
REPORT_COUNTRY_LABELS = 12
REPORT_SCATTER_SAMPLE = 8000
REPORT_US_COUNTRY = 'United States'
REPORT_US_TOP_FEATURES = 12
GLOBAL_MAIN_VIEW = 'Global main model'
US_REFIT_VIEW = 'United States only refit'

# Geography groups for the plain English median baseline
BASELINE_GROUP_SETS = [
    ['country_clean'],
    ['region'],
    []
]

# Default and notebook selected LightGBM presets for the canonical compensation runs
DEFAULT_LGB_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'l1',
    'boosting_type': 'gbdt',
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': -1
}
NOTEBOOK_TUNED_LGB_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'l1',
    'boosting_type': 'gbdt',
    'n_estimators': 659,
    'learning_rate': 0.04280768609535672,
    'num_leaves': 104,
    'max_depth': 11,
    'min_child_samples': 22,
    'subsample': 0.6872084049537491,
    'colsample_bytree': 0.7368883480938272,
    'reg_alpha': 9.28733327190043,
    'reg_lambda': 3.8666260444964735e-07,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': -1
}
LGB_PRESETS = {
    'default': DEFAULT_LGB_PARAMS,
    'tuned': NOTEBOOK_TUNED_LGB_PARAMS
}
SELECTED_LGB_PRESET = 'tuned'

# Numeric columns that need coercion before any downstream modeling work
NUMERIC_FIELDS = [
    'survey_year',
    'age_mid',
    'professional_experience_years',
    'language_count',
    'database_count',
    'platform_count',
    'comp_usd_clean',
    TARGET_COL,
    REAL_TARGET_COL,
    WINSOR_TARGET_COL,
    REAL_WINSOR_COL
]


# -------------------------------------------------------------------------------------
# Frame prep
# -------------------------------------------------------------------------------------

def add_compensation_derived_fields(frame):
    out = frame.copy()
    if 'role_family' not in out.columns and 'dev_type' in out.columns:
        out['role_family'] = out['dev_type'].map(comp_clean.build_role_family_value)
    out = comp_clean.add_multiselect_counts(
        out,
        fields=[
            'language',
            'database',
            'platform',
            'webframe',
            'misc_tech',
            'learn_code',
            'learn_code_online',
            'coding_activities',
            'op_sys_prof'
        ]
    )
    out = comp_clean.add_role_family_features(out, add_flags=True, add_count=False)
    return comp_clean.add_log_comp_real(out)


# Coerces the core numeric fields so downstream models don't trip over mixed dtypes
def coerce_numeric_fields(frame, fields=None):
    out = frame.copy()
    fields = NUMERIC_FIELDS if fields is None else fields

    for field in fields:
        if field in out.columns:
            out[field] = pd.to_numeric(out[field], errors='coerce')

    role_cols = [col for col in out.columns if col.startswith('role_') and col != 'role_family']
    for field in role_cols:
        out[field] = pd.to_numeric(out[field], errors='coerce')

    return out


# Adds a string version of survey year for categorical tree models
def add_survey_year_str(frame):
    out = frame.copy()
    out['survey_year_str'] = out['survey_year'].astype('string')
    return out


# Gets the compensation frame into one consistent modeling friendly shape
def coerce_compensation_frame(clean_core):
    frame = add_compensation_derived_fields(clean_core)
    frame = add_survey_year_str(frame)
    numeric_fields = NUMERIC_FIELDS + [col for col in frame.columns if col.startswith('role_') and col != 'role_family']
    frame = coerce_numeric_fields(frame, numeric_fields)

    if 'survey_year_str' in frame.columns:
        frame['survey_year_str'] = frame['survey_year_str'].astype(object)

    text_cols = frame.select_dtypes(include=['object', 'string']).columns
    for col in text_cols:
        frame.loc[frame[col].isna(), col] = np.nan

    return frame


# Carves the cleaned table into the core, tech rich, and AI era compensation windows
def get_comp_frames(clean_core):
    base = coerce_compensation_frame(clean_core)
    base = base.loc[
        comp_clean.comp_model_mask(base)
        & base['country_clean'].notna()
        & base['region'].notna()
    ].copy()

    core_df = base.copy()
    tech_df = base.loc[base['survey_year'].ge(min(TECH_WINDOW_YEARS))].copy()
    ai_df = base.loc[base['survey_year'].ge(min(AI_WINDOW_YEARS))].copy()
    return core_df, tech_df, ai_df


# Handy wrapper when we only want the canonical core compensation frame
def prepare_compensation_frame(clean_core):
    core_df, _, _ = get_comp_frames(clean_core)
    return core_df.copy()


# Pulls every role flag so the feature spec can stay readable elsewhere
def get_role_cols(frame):
    return sorted(col for col in frame.columns if col.startswith('role_') and col != 'role_family')


# Defines the core and later wave feature sets in one auditable place
def build_feature_sets(role_cols):
    core_cat = [
        'survey_year_str',
        'region',
        'country_clean',
        'employment_primary',
        'education_clean',
        'org_size_clean'
    ]
    core_num = [
        'age_mid',
        'professional_experience_years',
        'language_count',
        'database_count',
        'platform_count'
    ] + role_cols

    tech_cat = core_cat.copy()
    tech_num = core_num + ['learn_code_count', 'webframe_count', 'misc_tech_count']

    ai_cat = core_cat + ['remote_group', 'ai_use', 'ai_sent']
    ai_num = tech_num + ['learn_code_online_count', 'coding_activities_count', 'op_sys_prof_count']

    return {
        'core_cat': core_cat,
        'core_num': core_num,
        'tech_cat': tech_cat,
        'tech_num': tech_num,
        'ai_cat': ai_cat,
        'ai_num': ai_num
    }


def build_compensation_bundle(clean_core):
    core_df, tech_df, ai_df = get_comp_frames(clean_core)
    role_cols = get_role_cols(core_df)
    feature_sets = build_feature_sets(role_cols)
    return {
        'core_df': core_df,
        'tech_df': tech_df,
        'ai_df': ai_df,
        'role_cols': role_cols,
        'feature_sets': feature_sets
    }


# -------------------------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------------------------

# Finds the most common tech tokens so we can turn them into stable boolean flags
def extract_top_techs(frame, column, n):
    tokens = (
        frame[column]
        .dropna()
        .astype(str)
        .str.split(';')
        .explode()
        .str.strip()
    )
    tokens = tokens.loc[tokens.ne('')]
    return tokens.value_counts().head(n).index.tolist()


# Normalizes tech names into safe column labels
def tech_flag_name(column, tech):
    safe = (
        tech.lower()
        .replace('++', '_plusplus_')
        .replace('+', '_plus_')
        .replace('#', '_sharp_')
        .replace('.', '_dot_')
        .replace('/', '_slash_')
        .replace('&', '_and_')
        .replace(' ', '_')
    )
    safe = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in safe)
    safe = '_'.join(part for part in safe.split('_') if part)
    return f'{column}_{safe}'


# Builds train fit top tech flags so we don't leak future token popularity backwards
def add_top_tech_flags(frame, top_n_map=None, fit_frame=None):
    top_n_map = TOP_N_TECH if top_n_map is None else top_n_map
    fit_source = frame if fit_frame is None else fit_frame
    out = frame.copy()
    created = []

    for column, n in top_n_map.items():
        if column not in out.columns or column not in fit_source.columns:
            continue

        normalized = (
            ';'
            + out[column]
            .fillna('')
            .astype(str)
            .str.replace(r'\s*;\s*', ';', regex=True)
            .str.strip('; ')
            + ';'
        )

        for tech in extract_top_techs(fit_source, column, n):
            col_name = tech_flag_name(column, tech)
            out[col_name] = normalized.str.contains(f';{tech};', regex=False, na=False).astype(int)
            created.append(col_name)

    return out, created


# Splits a window and adds the same train fit tech flags to each split
def split_window_with_tech_flags(frame, train_years, valid_year, test_year, top_n_map=None):
    train_df, valid_df, test_df = model_audit.split_years(frame, train_years, valid_year, test_year)
    train_df, tech_flag_cols = add_top_tech_flags(train_df, top_n_map=top_n_map, fit_frame=train_df)
    valid_df, _ = add_top_tech_flags(valid_df, top_n_map=top_n_map, fit_frame=train_df)
    test_df, _ = add_top_tech_flags(test_df, top_n_map=top_n_map, fit_frame=train_df)
    return train_df, valid_df, test_df, tech_flag_cols


# Creates a train fit winsorized target for the upper tail compensation experiments
def add_winsor_targets(
    frame,
    fit_frame=None,
    group_cols=('country_clean',),
    lower_q=0.01,
    upper_q=0.99,
    min_group_size=20
):
    out = coerce_numeric_fields(frame, [REAL_TARGET_COL])
    fit_source = out if fit_frame is None else coerce_numeric_fields(fit_frame, [REAL_TARGET_COL])

    if group_cols:
        thresholds = (
            fit_source
            .groupby(list(group_cols), dropna=False)[REAL_TARGET_COL]
            .agg(
                group_size='size',
                lower=lambda series: series.quantile(lower_q),
                upper=lambda series: series.quantile(upper_q)
            )
            .reset_index()
        )
        thresholds.loc[thresholds['group_size'] < min_group_size, ['lower', 'upper']] = np.nan
        out = out.merge(thresholds[list(group_cols) + ['lower', 'upper']], on=list(group_cols), how='left')
    else:
        out['lower'] = np.nan
        out['upper'] = np.nan

    global_lower = fit_source[REAL_TARGET_COL].quantile(lower_q)
    global_upper = fit_source[REAL_TARGET_COL].quantile(upper_q)
    out['lower'] = out['lower'].fillna(global_lower)
    out['upper'] = out['upper'].fillna(global_upper)
    out[REAL_WINSOR_COL] = out[REAL_TARGET_COL].clip(lower=out['lower'], upper=out['upper'])
    out[WINSOR_TARGET_COL] = np.log(out[REAL_WINSOR_COL])
    return out.drop(columns=['lower', 'upper'])


# -------------------------------------------------------------------------------------
# Baselines and Ridge
# -------------------------------------------------------------------------------------

# Makes the feature frame play nicely with sklearn pipelines and imputers
def normalize_feature_frame(frame, cat_cols, num_cols):
    out = frame.copy()

    for col in cat_cols:
        out[col] = out[col].astype(object)
        out.loc[out[col].isna(), col] = np.nan

    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors='coerce').astype(float)

    return out


# Straightforward one hot plus scale preprocessing for the Ridge baseline
def build_ridge_pipe(cat_cols, num_cols):
    prep = ColumnTransformer([
        (
            'cat',
            Pipeline([
                ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
                ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=500))
            ]),
            cat_cols
        ),
        (
            'num',
            Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler())
            ]),
            num_cols
        )
    ])
    return Pipeline([
        ('prep', prep),
        ('model', Ridge())
    ])


# Geography median is the plain English benchmark we expect every real model to beat
def score_holdout_baseline(train_df, valid_df, test_df, group_sets=None):
    group_sets = BASELINE_GROUP_SETS if group_sets is None else group_sets
    valid_metrics = model_audit.score_hier_median(train_df, valid_df, group_sets)
    train_valid = pd.concat([train_df, valid_df], axis=0)
    test_metrics = model_audit.score_hier_median(train_valid, test_df, group_sets)
    return {
        'valid_metrics': valid_metrics,
        'test_metrics': test_metrics
    }


# Fits the interpretable linear baseline before we trust the tree models
def fit_ridge_holdout(train_df, valid_df, test_df, cat_cols, num_cols, target_col=TARGET_COL, alphas=None):
    alphas = [1.0, 5.0, 10.0, 25.0] if alphas is None else alphas
    feature_cols = cat_cols + num_cols
    train_df = normalize_feature_frame(train_df, cat_cols, num_cols)
    valid_df = normalize_feature_frame(valid_df, cat_cols, num_cols)
    test_df = normalize_feature_frame(test_df, cat_cols, num_cols)

    pipe = build_ridge_pipe(cat_cols, num_cols)
    best_alpha = None
    best_valid = None

    for alpha in alphas:
        candidate = clone(pipe)
        candidate.set_params(model=Ridge(alpha=alpha))
        candidate.fit(train_df[feature_cols], train_df[target_col])
        valid_metrics = model_audit.score_log_comp(valid_df[target_col], candidate.predict(valid_df[feature_cols]))

        if best_valid is None or valid_metrics['medae_real'] < best_valid['medae_real']:
            best_valid = valid_metrics
            best_alpha = alpha

    final_model = clone(pipe)
    final_model.set_params(model=Ridge(alpha=best_alpha))
    train_valid = pd.concat([train_df, valid_df], axis=0)
    final_model.fit(train_valid[feature_cols], train_valid[target_col])
    test_metrics = model_audit.score_log_comp(test_df[target_col], final_model.predict(test_df[feature_cols]))

    return {
        'model': final_model,
        'best_params': {'alpha': best_alpha},
        'valid_metrics': best_valid,
        'test_metrics': test_metrics,
        'feature_cols': feature_cols
    }


# -------------------------------------------------------------------------------------
# LightGBM
# -------------------------------------------------------------------------------------

# Preps categorical and numeric fields for LightGBM without changing the train/test contract
def prepare_lgbm_frame(frame, cat_cols, num_cols, target_col=TARGET_COL):
    out = coerce_compensation_frame(frame)
    out = out.copy()

    for col in cat_cols:
        out[col] = out[col].astype('string').fillna('Missing').astype('category')

    num_imputer = SimpleImputer(strategy='median')
    out_num = pd.DataFrame(
        num_imputer.fit_transform(out[num_cols]),
        columns=num_cols,
        index=out.index
    )
    out[num_cols] = out_num

    if target_col in out.columns:
        out[target_col] = pd.to_numeric(out[target_col], errors='coerce')

    return out, num_imputer


# Applies the saved numeric imputer so valid and test get the same treatment as train
def transform_lgbm_frame(frame, cat_cols, num_cols, num_imputer, target_col=TARGET_COL):
    out = coerce_compensation_frame(frame)
    out = out.copy()

    for col in cat_cols:
        out[col] = out[col].astype('string').fillna('Missing').astype('category')

    out_num = pd.DataFrame(
        num_imputer.transform(out[num_cols]),
        columns=num_cols,
        index=out.index
    )
    out[num_cols] = out_num

    if target_col in out.columns:
        out[target_col] = pd.to_numeric(out[target_col], errors='coerce')

    return out


# Resolves preset, manual overrides, and optional device choice in one place
def resolve_lgb_params(params=None, preset=None, device_type=None):
    if preset is None:
        preset = SELECTED_LGB_PRESET

    resolved = dict(LGB_PRESETS[preset])
    if params is not None:
        resolved.update(params)

    if device_type:
        resolved['device_type'] = device_type
    else:
        resolved.pop('device_type', None)

    return resolved


# Small Optuna tuner used for bounded local LightGBM searches
def tune_lightgbm(
    train_df,
    valid_df,
    cat_cols,
    num_cols,
    target_col=TARGET_COL,
    base_params=None,
    n_trials=20,
    device_type=None
):
    train_prepped, num_imputer = prepare_lgbm_frame(train_df, cat_cols, num_cols, target_col=target_col)
    valid_prepped = transform_lgbm_frame(valid_df, cat_cols, num_cols, num_imputer, target_col=target_col)
    features = cat_cols + num_cols
    base_params = resolve_lgb_params(base_params, preset='default', device_type=device_type)

    def objective(trial):
        params = dict(base_params)
        params.update({
            'n_estimators': trial.suggest_int('n_estimators', 200, 900),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 5, 16),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
        })
        model = lgb.LGBMRegressor(**params)
        model.fit(
            train_prepped[features],
            train_prepped[target_col].to_numpy(),
            eval_set=[(valid_prepped[features], valid_prepped[target_col].to_numpy())],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        pred_valid = model.predict(valid_prepped[features], num_iteration=model.best_iteration_)
        return model_audit.score_log_comp(valid_prepped[target_col], pred_valid)['medae_real']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = dict(base_params)
    best_params.update(study.best_trial.params)
    return best_params


# Standard single stage LightGBM regression fit on the year aware holdout contract
def fit_lightgbm_holdout(
    train_df,
    valid_df,
    test_df,
    cat_cols,
    num_cols,
    target_col=TARGET_COL,
    params=None,
    preset=None,
    tune_trials=0,
    device_type=None,
    early_stopping_rounds=50
):
    resolved_params = resolve_lgb_params(params, preset=preset, device_type=device_type)
    if tune_trials:
        resolved_params = tune_lightgbm(
            train_df,
            valid_df,
            cat_cols,
            num_cols,
            target_col=target_col,
            base_params=resolved_params,
            n_trials=tune_trials,
            device_type=device_type
        )

    train_prepped, num_imputer = prepare_lgbm_frame(train_df, cat_cols, num_cols, target_col=target_col)
    valid_prepped = transform_lgbm_frame(valid_df, cat_cols, num_cols, num_imputer, target_col=target_col)
    features = cat_cols + num_cols

    early_model = lgb.LGBMRegressor(**resolved_params)
    early_model.fit(
        train_prepped[features],
        train_prepped[target_col].to_numpy(),
        eval_set=[(valid_prepped[features], valid_prepped[target_col].to_numpy())],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
    )
    best_iteration = early_model.best_iteration_ or resolved_params.get('n_estimators', DEFAULT_LGB_PARAMS['n_estimators'])
    valid_pred = early_model.predict(valid_prepped[features], num_iteration=best_iteration)
    valid_metrics = model_audit.score_log_comp(
        valid_prepped[target_col],
        valid_pred
    )

    final_params = dict(resolved_params)
    final_params['n_estimators'] = best_iteration
    final_model = lgb.LGBMRegressor(**final_params)
    full_train_raw = pd.concat([train_df, valid_df], axis=0)
    full_train, full_num_imputer = prepare_lgbm_frame(full_train_raw, cat_cols, num_cols, target_col=target_col)
    full_test = transform_lgbm_frame(test_df, cat_cols, num_cols, full_num_imputer, target_col=target_col)
    final_model.fit(full_train[features], full_train[target_col].to_numpy())
    test_pred = final_model.predict(full_test[features])
    test_metrics = model_audit.score_log_comp(
        full_test[target_col],
        test_pred
    )

    return {
        'model': final_model,
        'best_params': final_params,
        'valid_metrics': valid_metrics,
        'test_metrics': test_metrics,
        'feature_cols': features,
        'num_imputer': full_num_imputer,
        'valid_pred_log': valid_pred,
        'test_pred_log': test_pred
    }


# Variant that trains on a train fit winsorized target while still scoring on real compensation
def fit_lightgbm_winsor_holdout(
    train_df,
    valid_df,
    test_df,
    cat_cols,
    num_cols,
    params=None,
    preset=None,
    tune_trials=0,
    device_type=None,
    group_cols=('country_clean',),
    early_stopping_rounds=50
):
    resolved_params = resolve_lgb_params(params, preset=preset, device_type=device_type)
    train_winsor = add_winsor_targets(train_df, fit_frame=train_df, group_cols=group_cols)
    valid_winsor = add_winsor_targets(valid_df, fit_frame=train_df, group_cols=group_cols)

    if tune_trials:
        resolved_params = tune_lightgbm(
            train_winsor,
            valid_winsor,
            cat_cols,
            num_cols,
            target_col=WINSOR_TARGET_COL,
            base_params=resolved_params,
            n_trials=tune_trials,
            device_type=device_type
        )

    train_prepped, num_imputer = prepare_lgbm_frame(
        train_winsor,
        cat_cols,
        num_cols,
        target_col=WINSOR_TARGET_COL
    )
    valid_prepped = transform_lgbm_frame(
        valid_winsor,
        cat_cols,
        num_cols,
        num_imputer,
        target_col=WINSOR_TARGET_COL
    )
    features = cat_cols + num_cols

    early_model = lgb.LGBMRegressor(**resolved_params)
    early_model.fit(
        train_prepped[features],
        train_prepped[WINSOR_TARGET_COL].to_numpy(),
        eval_set=[(valid_prepped[features], valid_prepped[WINSOR_TARGET_COL].to_numpy())],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
    )
    best_iteration = early_model.best_iteration_ or resolved_params.get('n_estimators', DEFAULT_LGB_PARAMS['n_estimators'])
    valid_pred = early_model.predict(valid_prepped[features], num_iteration=best_iteration)
    valid_metrics = model_audit.score_log_comp(valid_winsor[TARGET_COL], valid_pred)

    full_train_raw = pd.concat([train_df, valid_df], axis=0)
    full_train_winsor = add_winsor_targets(full_train_raw, fit_frame=full_train_raw, group_cols=group_cols)
    full_test_winsor = add_winsor_targets(test_df, fit_frame=full_train_raw, group_cols=group_cols)
    final_params = dict(resolved_params)
    final_params['n_estimators'] = best_iteration
    final_model = lgb.LGBMRegressor(**final_params)
    final_train, full_num_imputer = prepare_lgbm_frame(
        full_train_winsor,
        cat_cols,
        num_cols,
        target_col=WINSOR_TARGET_COL
    )
    final_test = transform_lgbm_frame(
        full_test_winsor,
        cat_cols,
        num_cols,
        full_num_imputer,
        target_col=WINSOR_TARGET_COL
    )
    final_model.fit(final_train[features], final_train[WINSOR_TARGET_COL].to_numpy())
    test_pred = final_model.predict(final_test[features])
    test_metrics = model_audit.score_log_comp(full_test_winsor[TARGET_COL], test_pred)

    return {
        'model': final_model,
        'best_params': final_params,
        'valid_metrics': valid_metrics,
        'test_metrics': test_metrics,
        'feature_cols': features,
        'num_imputer': full_num_imputer,
        'valid_pred_log': valid_pred,
        'test_pred_log': test_pred
    }


# -------------------------------------------------------------------------------------
# Canonical workflows
# -------------------------------------------------------------------------------------

# Main compensation comparison table used to lock the canonical model choice
def compare_same_sample_setups(
    clean_core,
    lgb_params=None,
    lgb_preset=None,
    top_n_map=None,
    tune_trials=0,
    device_type=None
):
    bundle = build_compensation_bundle(clean_core)
    feature_sets = bundle['feature_sets']
    core_df = bundle['core_df']
    tech_df = bundle['tech_df']
    ai_df = bundle['ai_df']

    core_train, core_valid, core_test = model_audit.split_years(core_df, CORE_WINDOW_YEARS, VALID_YEAR, TEST_YEAR)
    baseline = score_holdout_baseline(core_train, core_valid, core_test)
    ridge = fit_ridge_holdout(core_train, core_valid, core_test, feature_sets['core_cat'], feature_sets['core_num'])
    lgb_core = fit_lightgbm_holdout(
        core_train,
        core_valid,
        core_test,
        feature_sets['core_cat'],
        feature_sets['core_num'],
        params=lgb_params,
        preset=lgb_preset,
        tune_trials=tune_trials,
        device_type=device_type
    )

    core_flag_train, core_flag_valid, core_flag_test, core_tech_cols = split_window_with_tech_flags(
        core_df,
        CORE_WINDOW_YEARS,
        VALID_YEAR,
        TEST_YEAR,
        top_n_map=top_n_map
    )
    core_lgb_top_tech = fit_lightgbm_holdout(
        core_flag_train,
        core_flag_valid,
        core_flag_test,
        feature_sets['core_cat'],
        feature_sets['core_num'] + core_tech_cols,
        params=lgb_params,
        preset=lgb_preset,
        tune_trials=tune_trials,
        device_type=device_type
    )
    core_lgb_winsor = fit_lightgbm_winsor_holdout(
        core_flag_train,
        core_flag_valid,
        core_flag_test,
        feature_sets['core_cat'],
        feature_sets['core_num'] + core_tech_cols,
        params=lgb_params,
        preset=lgb_preset,
        tune_trials=tune_trials,
        device_type=device_type
    )

    tech_train, tech_valid, tech_test, tech_tech_cols = split_window_with_tech_flags(
        tech_df,
        TECH_WINDOW_YEARS,
        VALID_YEAR,
        TEST_YEAR,
        top_n_map=top_n_map
    )
    tech_lgb = fit_lightgbm_holdout(
        tech_train,
        tech_valid,
        tech_test,
        feature_sets['tech_cat'],
        feature_sets['tech_num'] + tech_tech_cols,
        params=lgb_params,
        preset=lgb_preset,
        tune_trials=tune_trials,
        device_type=device_type
    )

    ai_train, ai_valid, ai_test, ai_tech_cols = split_window_with_tech_flags(
        ai_df,
        AI_WINDOW_YEARS,
        VALID_YEAR,
        TEST_YEAR,
        top_n_map=top_n_map
    )
    ai_lgb = fit_lightgbm_holdout(
        ai_train,
        ai_valid,
        ai_test,
        feature_sets['ai_cat'],
        feature_sets['ai_num'] + ai_tech_cols,
        params=lgb_params,
        preset=lgb_preset,
        tune_trials=tune_trials,
        device_type=device_type
    )

    summary = pd.DataFrame([
        {
            'setup': 'Country-region median baseline',
            'valid_medae_real': baseline['valid_metrics']['medae_real'],
            'test_medae_real': baseline['test_metrics']['medae_real'],
            'test_rmse_real': baseline['test_metrics']['rmse_real'],
            'test_r2_log': baseline['test_metrics']['r2_log']
        },
        {
            'setup': 'Ridge core',
            'valid_medae_real': ridge['valid_metrics']['medae_real'],
            'test_medae_real': ridge['test_metrics']['medae_real'],
            'test_rmse_real': ridge['test_metrics']['rmse_real'],
            'test_r2_log': ridge['test_metrics']['r2_log']
        },
        {
            'setup': 'LightGBM core',
            'valid_medae_real': lgb_core['valid_metrics']['medae_real'],
            'test_medae_real': lgb_core['test_metrics']['medae_real'],
            'test_rmse_real': lgb_core['test_metrics']['rmse_real'],
            'test_r2_log': lgb_core['test_metrics']['r2_log']
        },
        {
            'setup': 'LightGBM core + top tech flags',
            'valid_medae_real': core_lgb_top_tech['valid_metrics']['medae_real'],
            'test_medae_real': core_lgb_top_tech['test_metrics']['medae_real'],
            'test_rmse_real': core_lgb_top_tech['test_metrics']['rmse_real'],
            'test_r2_log': core_lgb_top_tech['test_metrics']['r2_log']
        },
        {
            'setup': 'LightGBM winsorized target + top tech flags',
            'valid_medae_real': core_lgb_winsor['valid_metrics']['medae_real'],
            'test_medae_real': core_lgb_winsor['test_metrics']['medae_real'],
            'test_rmse_real': core_lgb_winsor['test_metrics']['rmse_real'],
            'test_r2_log': core_lgb_winsor['test_metrics']['r2_log']
        },
        {
            'setup': 'LightGBM tech-rich window',
            'valid_medae_real': tech_lgb['valid_metrics']['medae_real'],
            'test_medae_real': tech_lgb['test_metrics']['medae_real'],
            'test_rmse_real': tech_lgb['test_metrics']['rmse_real'],
            'test_r2_log': tech_lgb['test_metrics']['r2_log']
        },
        {
            'setup': 'LightGBM AI-era window',
            'valid_medae_real': ai_lgb['valid_metrics']['medae_real'],
            'test_medae_real': ai_lgb['test_metrics']['medae_real'],
            'test_rmse_real': ai_lgb['test_metrics']['rmse_real'],
            'test_r2_log': ai_lgb['test_metrics']['r2_log']
        }
    ]).sort_values('valid_medae_real')

    return {
        'bundle': bundle,
        'summary': summary,
        'baseline': baseline,
        'ridge': ridge,
        'lightgbm_core': lgb_core,
        'lightgbm_core_top_tech': core_lgb_top_tech,
        'lightgbm_winsor_top_tech': core_lgb_winsor,
        'tech_lgb': tech_lgb,
        'ai_lgb': ai_lgb,
        'tech_flag_cols': {
            'core': core_tech_cols,
            'tech': tech_tech_cols,
            'ai': ai_tech_cols
        },
        'selected_main_setup': 'LightGBM winsorized target + top tech flags',
        'selected_main_result': core_lgb_winsor
    }


# Returns the locked compensation model fit and the feature columns used to train it
def fit_selected_compensation_model(
    clean_core,
    lgb_params=None,
    lgb_preset=None,
    top_n_map=None,
    tune_trials=0,
    device_type=None
):
    bundle = build_compensation_bundle(clean_core)
    feature_sets = bundle['feature_sets']
    core_train, core_valid, core_test, tech_flag_cols = split_window_with_tech_flags(
        bundle['core_df'],
        CORE_WINDOW_YEARS,
        VALID_YEAR,
        TEST_YEAR,
        top_n_map=top_n_map
    )

    result = fit_lightgbm_winsor_holdout(
        core_train,
        core_valid,
        core_test,
        feature_sets['core_cat'],
        feature_sets['core_num'] + tech_flag_cols,
        params=lgb_params,
        preset=lgb_preset,
        tune_trials=tune_trials,
        device_type=device_type
    )

    return {
        'setup': 'LightGBM winsorized target + top tech flags',
        'train_years': CORE_WINDOW_YEARS,
        'valid_year': VALID_YEAR,
        'test_year': TEST_YEAR,
        'cat_cols': feature_sets['core_cat'],
        'num_cols': feature_sets['core_num'] + tech_flag_cols,
        'tech_flag_cols': tech_flag_cols,
        'bundle': bundle,
        **result
    }


# Rolling origin helper used to check whether a setup holds up as the train window expands
def rolling_origin_lightgbm(
    frame,
    cat_cols,
    num_cols,
    target_col=TARGET_COL,
    params=None,
    preset=None,
    top_n_map=None,
    use_top_tech=False,
    use_winsor_target=False,
    min_train_year=2019,
    final_valid_year=2024,
    device_type=None,
    early_stopping_rounds=50
):
    rows = []

    for train_years, valid_year, train_df, valid_df in model_audit.rolling_origin_splits(
        frame,
        min_train_year=min_train_year,
        final_valid_year=final_valid_year
    ):
        train_work = train_df.copy()
        valid_work = valid_df.copy()
        tech_flag_cols = []

        if use_top_tech:
            train_work, tech_flag_cols = add_top_tech_flags(train_work, top_n_map=top_n_map, fit_frame=train_work)
            valid_work, _ = add_top_tech_flags(valid_work, top_n_map=top_n_map, fit_frame=train_work)

        num_cols_fold = num_cols + tech_flag_cols
        target_col_fold = target_col

        if use_winsor_target:
            fold_result = fit_lightgbm_winsor_holdout(
                train_work,
                valid_work,
                valid_work,
                cat_cols,
                num_cols_fold,
                params=params,
                preset=preset,
                tune_trials=0,
                device_type=device_type,
                group_cols=('country_clean',),
                early_stopping_rounds=early_stopping_rounds
            )
        else:
            fold_result = fit_lightgbm_holdout(
                train_work,
                valid_work,
                valid_work,
                cat_cols,
                num_cols_fold,
                target_col=target_col_fold,
                params=params,
                preset=preset,
                tune_trials=0,
                device_type=device_type,
                early_stopping_rounds=early_stopping_rounds
            )
        rows.append({
            'train_years': ','.join(map(str, train_years)),
            'valid_year': valid_year,
            **fold_result['valid_metrics']
        })

    return pd.DataFrame(rows)


# Compares the baseline and LightGBM families under rolling origin validation
def rolling_origin_setup_comparison(
    clean_core,
    ridge_alphas=None,
    lgb_presets=None,
    top_n_map=None,
    min_train_year=2019,
    final_valid_year=2024,
    device_type=None
):
    ridge_alphas = [25.0] if ridge_alphas is None else ridge_alphas
    lgb_presets = {'default': DEFAULT_LGB_PARAMS} if lgb_presets is None else lgb_presets

    bundle = build_compensation_bundle(clean_core)
    feature_sets = bundle['feature_sets']
    core_df = bundle['core_df']
    rows = []

    for alpha in ridge_alphas:
        fold_rows = []
        for train_years, valid_year, train_df, valid_df in model_audit.rolling_origin_splits(
            core_df,
            min_train_year=min_train_year,
            final_valid_year=final_valid_year
        ):
            ridge = fit_ridge_holdout(
                train_df,
                valid_df,
                valid_df,
                feature_sets['core_cat'],
                feature_sets['core_num'],
                alphas=[alpha]
            )
            fold_rows.append({
                'train_years': ','.join(map(str, train_years)),
                'valid_year': valid_year,
                **ridge['valid_metrics']
            })

        folds = pd.DataFrame(fold_rows)
        rows.append({
            'setup': f'Ridge core alpha={alpha}',
            'folds': folds
        })

    for preset_name, preset_params in lgb_presets.items():
        core_folds = rolling_origin_lightgbm(
            core_df,
            feature_sets['core_cat'],
            feature_sets['core_num'],
            params=preset_params,
            preset=None,
            min_train_year=min_train_year,
            final_valid_year=final_valid_year,
            device_type=device_type
        )
        rows.append({
            'setup': f'LightGBM core [{preset_name}]',
            'folds': core_folds
        })

        top_tech_folds = rolling_origin_lightgbm(
            core_df,
            feature_sets['core_cat'],
            feature_sets['core_num'],
            params=preset_params,
            preset=None,
            top_n_map=top_n_map,
            use_top_tech=True,
            min_train_year=min_train_year,
            final_valid_year=final_valid_year,
            device_type=device_type
        )
        rows.append({
            'setup': f'LightGBM core + top tech flags [{preset_name}]',
            'folds': top_tech_folds
        })

        winsor_folds = rolling_origin_lightgbm(
            core_df,
            feature_sets['core_cat'],
            feature_sets['core_num'],
            params=preset_params,
            preset=None,
            top_n_map=top_n_map,
            use_top_tech=True,
            use_winsor_target=True,
            min_train_year=min_train_year,
            final_valid_year=final_valid_year,
            device_type=device_type
        )
        rows.append({
            'setup': f'LightGBM winsorized target + top tech flags [{preset_name}]',
            'folds': winsor_folds
        })

    summary = pd.DataFrame([
        {
            'setup': row['setup'],
            'fold_count': len(row['folds']),
            'mean_valid_medae_real': row['folds']['medae_real'].mean(),
            'mean_valid_rmse_real': row['folds']['rmse_real'].mean(),
            'mean_valid_r2_log': row['folds']['r2_log'].mean()
        }
        for row in rows
    ]).sort_values('mean_valid_medae_real')

    return {
        'bundle': bundle,
        'results': rows,
        'summary': summary
    }


# -------------------------------------------------------------------------------------
# Reporting
# -------------------------------------------------------------------------------------

# Keeps saved reporting artifacts in one stable directory with predictable file names
def build_report_paths(output_dir=REPORT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        'output_dir': output_dir,
        'figures': {
            'context': output_dir / 'comp_context.png',
            'setup_compare': output_dir / 'comp_setup_compare.png',
            'winsor_vs_plain': output_dir / 'comp_winsor_vs_plain_by_decile.png',
            'test_diag': output_dir / 'comp_test_diagnostics.png',
            'subgroup_error': output_dir / 'comp_subgroup_error.png',
            'geo_alignment': output_dir / 'comp_geography_alignment.png',
            'shap_bar': output_dir / 'comp_shap_bar.png',
            'shap_beeswarm': output_dir / 'comp_shap_beeswarm.png',
            'shap_family': output_dir / 'comp_shap_family.png',
            'rolling_origin': output_dir / 'comp_rolling_origin.png'
        },
        'tables': {
            'setup_summary': output_dir / 'comp_setup_summary.csv',
            'split_counts': output_dir / 'comp_split_counts.csv',
            'region_metrics': output_dir / 'comp_test_region_metrics.csv',
            'country_metrics': output_dir / 'comp_test_country_metrics.csv',
            'decile_compare': output_dir / 'comp_test_decile_compare.csv',
            'feature_importance': output_dir / 'comp_feature_importance.csv',
            'shap_top_features': output_dir / 'comp_shap_top_features.csv',
            'shap_family': output_dir / 'comp_shap_family_importance.csv'
        },
        'manifest': output_dir / 'comp_report_manifest.json'
    }


# Keeps the United States side report artifacts in their own stable directory
def build_us_report_paths(output_dir=REPORT_US_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        'output_dir': output_dir,
        'figures': {
            'compare': output_dir / 'comp_us_compare.png',
            'diagnostics': output_dir / 'comp_us_diagnostics.png',
            'feature_shift': output_dir / 'comp_us_feature_shift.png',
            'shap_beeswarm': output_dir / 'comp_us_shap_beeswarm.png'
        },
        'tables': {
            'summary': output_dir / 'comp_us_summary.csv',
            'test_predictions': output_dir / 'comp_us_test_predictions.csv',
            'shap_compare': output_dir / 'comp_us_shap_compare.csv',
            'shap_top_features': output_dir / 'comp_us_shap_top_features.csv'
        },
        'manifest': output_dir / 'comp_us_report_manifest.json'
    }


# Converts raw dollar values into shorter axis labels that are easier to read on plots
def money_formatter(x, _pos):
    if np.isnan(x):
        return ''
    if abs(x) >= 1_000_000:
        return f'${x / 1_000_000:.1f}M'
    if abs(x) >= 1_000:
        return f'${x / 1_000:.0f}k'
    return f'${x:.0f}'


# Applies the shared dollar formatter so all money axes read the same way
def format_money_axis(ax, axis='x'):
    formatter = FuncFormatter(money_formatter)
    if axis in {'x', 'both'}:
        ax.xaxis.set_major_formatter(formatter)
    if axis in {'y', 'both'}:
        ax.yaxis.set_major_formatter(formatter)


# Builds a scored frame with real dollar predictions and residuals for diagnostics
def score_prediction_frame(frame, pred_log, actual_log_col=TARGET_COL):
    out = frame.copy()
    out['pred_log'] = pd.Series(pred_log, index=out.index).astype(float)
    out['actual_log'] = pd.to_numeric(out[actual_log_col], errors='coerce')
    out['pred_real'] = np.exp(out['pred_log'])
    out['actual_real'] = np.exp(out['actual_log'])
    out['signed_error_real'] = out['pred_real'] - out['actual_real']
    out['abs_error_real'] = out['signed_error_real'].abs()
    out['pct_error'] = np.where(
        out['actual_real'].gt(0),
        out['abs_error_real'] / out['actual_real'] * 100.0,
        np.nan
    )
    return out


# Flattens a scored prediction frame into one shared export schema for report tables
def build_scored_prediction_export(scored_df, model_view, fit_scope, test_scope, report_country=None):
    id_cols = [col for col in ['row_id', 'response_id', 'survey_year', 'country_clean', 'region'] if col in scored_df.columns]
    out = scored_df[id_cols].copy()
    out['model_view'] = model_view
    out['fit_scope'] = fit_scope
    out['test_scope'] = test_scope
    if report_country is not None and 'country_clean' in out.columns:
        out['is_report_country'] = out['country_clean'].eq(report_country)
    out['actual_real'] = scored_df['actual_real'].to_numpy()
    out['pred_real'] = scored_df['pred_real'].to_numpy()
    out['signed_error_real'] = scored_df['signed_error_real'].to_numpy()
    out['abs_error_real'] = scored_df['abs_error_real'].to_numpy()
    out['pct_error'] = scored_df['pct_error'].to_numpy()
    return out


# Summarizes where the model does well or poorly once the test rows are grouped
def summarize_prediction_groups(scored_df, group_cols, train_ref=None):
    rows = []

    for key, sub in scored_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = dict(zip(group_cols, key, strict=False))
        row['test_rows'] = len(sub)
        row['actual_median_real'] = float(sub['actual_real'].median())
        row['pred_median_real'] = float(sub['pred_real'].median())
        row['medae_real'] = float(sub['abs_error_real'].median())
        row['mae_real'] = float(sub['abs_error_real'].mean())
        row['rmse_real'] = float(np.sqrt(np.mean(sub['signed_error_real'] ** 2)))
        row['median_signed_error_real'] = float(sub['signed_error_real'].median())
        row['median_pct_error'] = float(sub['pct_error'].median())
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if train_ref is not None:
        support = train_ref.groupby(group_cols, dropna=False).size().rename('train_rows').reset_index()
        out = out.merge(support, on=group_cols, how='left')
        out['train_rows'] = out['train_rows'].fillna(0).astype(int)

    return out.sort_values('medae_real', ascending=False).reset_index(drop=True)


# Uses either shared or independent deciles to compare how model views behave by compensation band
def build_named_decile_compare_table(scored_frames, shared_deciles=True):
    labels = [f'D{idx}' for idx in range(1, 11)]
    decile_dtype = pd.CategoricalDtype(categories=labels, ordered=True)
    first_setup, first_scored = scored_frames[0]
    out_first = first_scored.copy()

    def assign_deciles(scored):
        out = scored.copy()
        bin_count = min(10, len(out))
        if bin_count < 2:
            out['actual_decile'] = pd.Categorical(['D1'] * len(out), categories=labels, ordered=True)
        else:
            out['actual_decile'] = pd.qcut(
                out['actual_real'].rank(method='first'),
                bin_count,
                labels=labels[:bin_count]
            ).astype(decile_dtype)
        return out

    out_first = assign_deciles(out_first)

    rows = []
    for setup_name, scored in [(first_setup, out_first)] + list(scored_frames[1:]):
        if setup_name == first_setup:
            working = out_first.copy()
        elif shared_deciles:
            working = scored.copy()
            working['actual_decile'] = pd.Categorical(
                out_first['actual_decile'].to_numpy(),
                categories=labels,
                ordered=True
            )
        else:
            working = assign_deciles(scored)
        grouped = (
            working
            .groupby('actual_decile', observed=False)
            .agg(
                rows=('actual_real', 'size'),
                median_actual_real=('actual_real', 'median'),
                medae_real=('abs_error_real', 'median'),
                mae_real=('abs_error_real', 'mean'),
                median_signed_error_real=('signed_error_real', 'median'),
                median_pct_error=('pct_error', 'median')
            )
            .reset_index()
        )
        grouped['actual_decile'] = grouped['actual_decile'].astype(decile_dtype)
        grouped['setup'] = setup_name
        rows.append(grouped)

    return (
        pd.concat(rows, axis=0)
        .sort_values(['setup', 'actual_decile'])
        .reset_index(drop=True)
    )


# Uses the same test rows to compare where the winsorized and plain models diverge
def build_decile_compare_table(selected_scored, plain_scored):
    return build_named_decile_compare_table([
        ('LightGBM core + top tech flags', plain_scored),
        ('LightGBM winsorized target + top tech flags', selected_scored)
    ], shared_deciles=True)


# Maps raw feature names into broader story buckets for presentation ready importance views
def feature_family(feature):
    if feature in {'country_clean', 'region'}:
        return 'Geography'
    if feature == 'survey_year_str':
        return 'Survey year'
    if feature == 'employment_primary':
        return 'Employment'
    if feature == 'education_clean':
        return 'Education'
    if feature == 'org_size_clean':
        return 'Organization size'
    if feature in {'age_mid', 'professional_experience_years'}:
        return 'Experience and age'
    if feature in {'language_count', 'database_count', 'platform_count'}:
        return 'Tech breadth'
    if feature.startswith('language_') or feature.startswith('database_') or feature.startswith('platform_'):
        return 'Top tech flags'
    if feature.startswith('role_'):
        return 'Role'
    return 'Other'


# Rolls feature level importance up into broader families so the hidden structure is easier to see
def aggregate_feature_family_importance(feature_df, value_col):
    out = feature_df.copy()
    out['family'] = out['feature'].map(feature_family)
    return (
        out
        .groupby('family', as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
        .reset_index(drop=True)
    )


# Converts raw importance magnitudes into within view shares for comparison plots
def normalize_importance_values(feature_df, value_col='mean_abs_shap'):
    out = feature_df.copy()
    total = out[value_col].sum()
    out[f'normalized_{value_col}'] = np.where(total > 0, out[value_col] / total, 0.0)
    return out.sort_values(value_col, ascending=False).reset_index(drop=True)


# SHAP gives the cleanest feature driver view for the locked LightGBM model
def build_selected_shap_bundle(model, feature_frame, sample_size=REPORT_SHAP_SAMPLE):
    shap_frame = feature_frame.copy()
    if sample_size is not None and len(shap_frame) > sample_size:
        shap_frame = shap_frame.sample(sample_size, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_frame)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    display_frame = shap_frame.copy()
    for col in display_frame.columns:
        if str(display_frame[col].dtype) == 'category':
            display_frame[col] = display_frame[col].astype(str)

    top_features = pd.DataFrame({
        'feature': feature_frame.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    return {
        'feature_frame': shap_frame,
        'display_frame': display_frame,
        'shap_values': shap_values,
        'top_features': top_features,
        'family_importance': aggregate_feature_family_importance(top_features, 'mean_abs_shap')
    }


# Pulls together the model, predictions, SHAP, and subgroup summaries needed for reporting
def build_compensation_report_bundle(
    clean_core,
    lgb_params=None,
    lgb_preset=None,
    top_n_map=None,
    shap_sample_size=REPORT_SHAP_SAMPLE,
    include_shap=True,
    include_rolling=True
):
    comparison = compare_same_sample_setups(
        clean_core,
        lgb_params=lgb_params,
        lgb_preset=lgb_preset,
        top_n_map=top_n_map
    )
    bundle = comparison['bundle']
    feature_sets = bundle['feature_sets']
    core_df = bundle['core_df']

    core_train, core_valid, core_test, tech_flag_cols = split_window_with_tech_flags(
        core_df,
        CORE_WINDOW_YEARS,
        VALID_YEAR,
        TEST_YEAR,
        top_n_map=top_n_map
    )
    train_valid = pd.concat([core_train, core_valid], axis=0)

    selected_result = comparison['selected_main_result']
    plain_result = comparison['lightgbm_core_top_tech']

    selected_test = add_winsor_targets(core_test, fit_frame=train_valid, group_cols=('country_clean',))
    selected_test_prepped = transform_lgbm_frame(
        selected_test,
        feature_sets['core_cat'],
        feature_sets['core_num'] + tech_flag_cols,
        selected_result['num_imputer'],
        target_col=WINSOR_TARGET_COL
    )
    selected_pred_log = selected_result['model'].predict(selected_test_prepped[selected_result['feature_cols']])
    selected_scored = score_prediction_frame(selected_test, selected_pred_log, actual_log_col=TARGET_COL)

    plain_test_prepped = transform_lgbm_frame(
        core_test,
        feature_sets['core_cat'],
        feature_sets['core_num'] + tech_flag_cols,
        plain_result['num_imputer'],
        target_col=TARGET_COL
    )
    plain_pred_log = plain_result['model'].predict(plain_test_prepped[plain_result['feature_cols']])
    plain_scored = score_prediction_frame(core_test, plain_pred_log, actual_log_col=TARGET_COL)

    split_counts = (
        core_df
        .assign(
            split=lambda df: np.where(
                df['survey_year'].eq(TEST_YEAR),
                'test',
                np.where(df['survey_year'].eq(VALID_YEAR), 'valid', 'train')
            )
        )
        .groupby(['survey_year', 'split'], as_index=False)
        .size()
        .rename(columns={'size': 'rows'})
    )

    region_metrics = summarize_prediction_groups(selected_scored, ['region'], train_ref=train_valid)
    country_metrics = summarize_prediction_groups(selected_scored, ['country_clean', 'region'], train_ref=train_valid)
    decile_compare = build_decile_compare_table(selected_scored, plain_scored)

    feature_importance = pd.DataFrame({
        'feature': selected_result['feature_cols'],
        'importance': selected_result['model'].feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    shap_bundle = None
    if include_shap:
        shap_bundle = build_selected_shap_bundle(
            selected_result['model'],
            selected_test_prepped[selected_result['feature_cols']],
            sample_size=shap_sample_size
        )

    rolling = None
    if include_rolling:
        rolling = rolling_origin_setup_comparison(
            clean_core,
            ridge_alphas=[25.0],
            lgb_presets={SELECTED_LGB_PRESET: LGB_PRESETS[SELECTED_LGB_PRESET]},
            top_n_map=top_n_map
        )

    return {
        'comparison': comparison,
        'bundle': bundle,
        'core_train': core_train,
        'core_valid': core_valid,
        'core_test': core_test,
        'train_valid': train_valid,
        'split_counts': split_counts,
        'selected_result': selected_result,
        'plain_result': plain_result,
        'selected_scored': selected_scored,
        'plain_scored': plain_scored,
        'region_metrics': region_metrics,
        'country_metrics': country_metrics,
        'decile_compare': decile_compare,
        'feature_importance': feature_importance,
        'shap_bundle': shap_bundle,
        'rolling': rolling
    }


# Builds a United States side report that compares the global model view to a US only refit
def build_us_compensation_report_bundle(
    clean_core,
    country=REPORT_US_COUNTRY,
    lgb_params=None,
    lgb_preset=None,
    top_n_map=None,
    shap_sample_size=REPORT_SHAP_SAMPLE,
    include_shap=True
):
    bundle = build_compensation_bundle(clean_core)
    feature_sets = bundle['feature_sets']
    core_df = bundle['core_df']
    country_df = core_df.loc[core_df['country_clean'].eq(country)].copy()

    if country_df.empty:
        raise ValueError(f"No comparable compensation rows found for {country}")

    global_train, global_valid, global_test, global_tech_cols = split_window_with_tech_flags(
        core_df,
        CORE_WINDOW_YEARS,
        VALID_YEAR,
        TEST_YEAR,
        top_n_map=top_n_map
    )
    global_result = fit_lightgbm_winsor_holdout(
        global_train,
        global_valid,
        global_test,
        feature_sets['core_cat'],
        feature_sets['core_num'] + global_tech_cols,
        params=lgb_params,
        preset=lgb_preset,
        tune_trials=0,
        group_cols=('country_clean',)
    )
    global_test_scored = score_prediction_frame(
        global_test,
        pd.Series(global_result['test_pred_log'], index=global_test.index),
        actual_log_col=TARGET_COL
    )

    country_train, country_valid, country_test, country_tech_cols = split_window_with_tech_flags(
        country_df,
        CORE_WINDOW_YEARS,
        VALID_YEAR,
        TEST_YEAR,
        top_n_map=top_n_map
    )
    country_cat_cols = [col for col in feature_sets['core_cat'] if col not in ['country_clean', 'region']]
    country_num_cols = feature_sets['core_num'] + country_tech_cols
    country_result = fit_lightgbm_winsor_holdout(
        country_train,
        country_valid,
        country_test,
        country_cat_cols,
        country_num_cols,
        params=lgb_params,
        preset=lgb_preset,
        tune_trials=0,
        group_cols=()
    )
    country_test_scored = score_prediction_frame(
        country_test,
        pd.Series(country_result['test_pred_log'], index=country_test.index),
        actual_log_col=TARGET_COL
    )

    summary = pd.DataFrame([
        {
            'model_view': GLOBAL_MAIN_VIEW,
            'fit_scope': 'global',
            'country': 'All countries',
            'train_rows': int(len(global_train)),
            'valid_rows': int(len(global_valid)),
            'test_rows': int(len(global_test)),
            'feature_count': int(len(global_result['feature_cols'])),
            'valid_medae_real': float(global_result['valid_metrics']['medae_real']),
            'test_medae_real': float(global_result['test_metrics']['medae_real']),
            'test_rmse_real': float(global_result['test_metrics']['rmse_real']),
            'test_r2_log': float(global_result['test_metrics']['r2_log'])
        },
        {
            'model_view': US_REFIT_VIEW,
            'fit_scope': 'country_only',
            'country': country,
            'train_rows': int(len(country_train)),
            'valid_rows': int(len(country_valid)),
            'test_rows': int(len(country_test)),
            'feature_count': int(len(country_result['feature_cols'])),
            'valid_medae_real': float(country_result['valid_metrics']['medae_real']),
            'test_medae_real': float(country_result['test_metrics']['medae_real']),
            'test_rmse_real': float(country_result['test_metrics']['rmse_real']),
            'test_r2_log': float(country_result['test_metrics']['r2_log'])
        }
    ])

    split_counts = (
        country_df
        .assign(
            split=lambda df: np.where(
                df['survey_year'].eq(TEST_YEAR),
                'test',
                np.where(df['survey_year'].eq(VALID_YEAR), 'valid', 'train')
            )
        )
        .groupby(['survey_year', 'split'], as_index=False)
        .size()
        .rename(columns={'size': 'rows'})
    )

    country_scope = f"{country.lower().replace(' ', '_')}_test_{TEST_YEAR}"
    predictions = pd.concat([
        build_scored_prediction_export(
            global_test_scored,
            model_view=GLOBAL_MAIN_VIEW,
            fit_scope='global',
            test_scope=f'global_test_{TEST_YEAR}',
            report_country=country
        ),
        build_scored_prediction_export(
            country_test_scored,
            model_view=US_REFIT_VIEW,
            fit_scope='country_only',
            test_scope=country_scope,
            report_country=country
        )
    ], axis=0).reset_index(drop=True)

    decile_compare = build_named_decile_compare_table([
        (GLOBAL_MAIN_VIEW, global_test_scored),
        (US_REFIT_VIEW, country_test_scored)
    ], shared_deciles=False)

    global_shap_bundle = None
    country_shap_bundle = None
    shap_compare = None
    shap_top_features = None

    if include_shap:
        global_train_valid = pd.concat([global_train, global_valid], axis=0)
        global_test_winsor = add_winsor_targets(global_test, fit_frame=global_train_valid, group_cols=('country_clean',))
        global_test_prepped = transform_lgbm_frame(
            global_test_winsor,
            feature_sets['core_cat'],
            feature_sets['core_num'] + global_tech_cols,
            global_result['num_imputer'],
            target_col=WINSOR_TARGET_COL
        )

        country_train_valid = pd.concat([country_train, country_valid], axis=0)
        country_test_winsor = add_winsor_targets(country_test, fit_frame=country_train_valid, group_cols=())
        country_test_prepped = transform_lgbm_frame(
            country_test_winsor,
            country_cat_cols,
            country_num_cols,
            country_result['num_imputer'],
            target_col=WINSOR_TARGET_COL
        )

        global_feature_frame = global_test_prepped.loc[:, global_result['feature_cols']].copy()
        country_feature_frame = country_test_prepped.loc[:, country_result['feature_cols']].copy()

        global_shap_bundle = build_selected_shap_bundle(
            global_result['model'],
            global_feature_frame,
            sample_size=shap_sample_size
        )
        country_shap_bundle = build_selected_shap_bundle(
            country_result['model'],
            country_feature_frame,
            sample_size=shap_sample_size
        )

        global_shap_compare = normalize_importance_values(global_shap_bundle['top_features'], value_col='mean_abs_shap')
        global_shap_compare['model_view'] = GLOBAL_MAIN_VIEW
        country_shap_compare = normalize_importance_values(country_shap_bundle['top_features'], value_col='mean_abs_shap')
        country_shap_compare['model_view'] = US_REFIT_VIEW
        shap_compare = pd.concat([global_shap_compare, country_shap_compare], axis=0).reset_index(drop=True)
        shap_top_features = country_shap_bundle['top_features'].copy()

    return {
        'country': country,
        'bundle': bundle,
        'country_df': country_df,
        'country_train': country_train,
        'country_valid': country_valid,
        'country_test': country_test,
        'summary': summary,
        'split_counts': split_counts,
        'global_result': global_result,
        'country_result': country_result,
        'global_test_scored': global_test_scored,
        'country_test_scored': country_test_scored,
        'predictions': predictions,
        'decile_compare': decile_compare,
        'global_shap_bundle': global_shap_bundle,
        'country_shap_bundle': country_shap_bundle,
        'shap_compare': shap_compare,
        'shap_top_features': shap_top_features
    }


# Context figure that explains the sample window and why the log target was used
def plot_compensation_context(report_bundle, path):
    core_df = report_bundle['bundle']['core_df']
    split_counts = report_bundle['split_counts']
    comp_real = pd.to_numeric(core_df[REAL_TARGET_COL], errors='coerce')
    comp_real = comp_real[comp_real.gt(0)]
    comp_bins = 40
    if len(comp_real) > 1 and comp_real.min() < comp_real.max():
        comp_bins = np.logspace(np.log10(comp_real.min()), np.log10(comp_real.max()), 40)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    sns.barplot(data=split_counts, x='survey_year', y='rows', hue='split', ax=axes[0], palette='Set2')
    axes[0].set_title('Compensation Sample Rows By Year')
    axes[0].set_xlabel('Survey year')
    axes[0].set_ylabel('Rows')
    axes[0].legend(title='Split')

    sns.histplot(comp_real, bins=comp_bins, ax=axes[1], color='#4C78A8', edgecolor='white', linewidth=0.3)
    axes[1].set_xscale('log')
    axes[1].set_title('Cleaned Compensation Distribution')
    axes[1].set_xlabel('Compensation in 2025 USD')
    axes[1].set_ylabel('Rows')
    format_money_axis(axes[1], axis='x')

    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# Main setup figure that shows why the selected model won the comparison
def plot_setup_comparison(report_bundle, path):
    summary = report_bundle['comparison']['summary'].copy()
    summary['is_selected'] = summary['setup'].eq(report_bundle['comparison']['selected_main_setup'])
    palette = ['#E45756' if flag else '#72B7B2' for flag in summary['is_selected']]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    sns.barplot(data=summary, x='valid_medae_real', y='setup', ax=axes[0], palette=palette, orient='h')
    axes[0].set_title('Validation MedAE By Setup')
    axes[0].set_xlabel('Validation MedAE in 2025 USD')
    axes[0].set_ylabel('')
    format_money_axis(axes[0], axis='x')

    sns.barplot(data=summary, x='test_medae_real', y='setup', ax=axes[1], palette=palette, orient='h')
    axes[1].set_title('Test MedAE By Setup')
    axes[1].set_xlabel('Test MedAE in 2025 USD')
    axes[1].set_ylabel('')
    format_money_axis(axes[1], axis='x')

    sns.barplot(data=summary, x='test_rmse_real', y='setup', ax=axes[2], palette=palette, orient='h')
    axes[2].set_title('Test RMSE By Setup')
    axes[2].set_xlabel('Test RMSE in 2025 USD')
    axes[2].set_ylabel('')
    format_money_axis(axes[2], axis='x')

    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# Shows where the winsorized target helped and where it stayed basically the same
def plot_winsor_vs_plain_by_decile(report_bundle, path):
    decile_compare = report_bundle['decile_compare'].sort_values(['setup', 'actual_decile']).copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.lineplot(
        data=decile_compare,
        x='actual_decile',
        y='medae_real',
        hue='setup',
        marker='o',
        ax=axes[0]
    )
    axes[0].set_title('Test MedAE By Actual Compensation Decile')
    axes[0].set_xlabel('Actual compensation decile')
    axes[0].set_ylabel('Median absolute error in 2025 USD')
    format_money_axis(axes[0], axis='y')

    sns.lineplot(
        data=decile_compare,
        x='actual_decile',
        y='median_signed_error_real',
        hue='setup',
        marker='o',
        ax=axes[1]
    )
    axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[1].set_title('Median Signed Error By Actual Compensation Decile')
    axes[1].set_xlabel('Actual compensation decile')
    axes[1].set_ylabel('Prediction minus actual in 2025 USD')
    format_money_axis(axes[1], axis='y')

    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend_.remove()
    axes[1].legend(handles, labels, title='Setup')
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# Test set diagnostics that show the main fit and the remaining residual spread
def plot_test_diagnostics(report_bundle, path):
    scored = report_bundle['selected_scored']
    diag_min = min(scored['actual_real'].min(), scored['pred_real'].min())
    diag_max = max(scored['actual_real'].max(), scored['pred_real'].max())
    diag_df = scored[['actual_real', 'pred_real']].dropna()
    if len(diag_df) > REPORT_SCATTER_SAMPLE:
        diag_df = diag_df.sample(REPORT_SCATTER_SAMPLE, random_state=RANDOM_STATE)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(
        diag_df['actual_real'],
        diag_df['pred_real'],
        s=12,
        alpha=0.18,
        color='#4C78A8',
        edgecolors='none',
        rasterized=len(diag_df) > 2000
    )
    axes[0].plot([diag_min, diag_max], [diag_min, diag_max], color='#F58518', linestyle='--', linewidth=1.5)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlim(diag_min * 0.95, diag_max * 1.05)
    axes[0].set_ylim(diag_min * 0.95, diag_max * 1.05)
    axes[0].set_title('Actual vs Predicted Compensation on Test 2025')
    axes[0].set_xlabel('Actual compensation in 2025 USD')
    axes[0].set_ylabel('Predicted compensation in 2025 USD')
    format_money_axis(axes[0], axis='both')

    sns.histplot(scored['signed_error_real'], bins=40, ax=axes[1], color='#72B7B2')
    axes[1].axvline(0, color='black', linewidth=1, linestyle='--')
    axes[1].set_title('Signed Error Distribution on Test 2025')
    axes[1].set_xlabel('Prediction minus actual in 2025 USD')
    axes[1].set_ylabel('Rows')
    format_money_axis(axes[1], axis='x')

    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# Geography dominates compensation, so this figure shows where errors are concentrated
def plot_subgroup_error(report_bundle, path):
    region_metrics = report_bundle['region_metrics'].sort_values('medae_real', ascending=False)
    country_metrics = report_bundle['country_metrics'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    sns.barplot(data=region_metrics, x='medae_real', y='region', ax=axes[0], palette='crest')
    axes[0].set_title('Region-Level Test MedAE')
    axes[0].set_xlabel('Median absolute error in 2025 USD')
    axes[0].set_ylabel('')
    format_money_axis(axes[0], axis='x')

    sns.scatterplot(
        data=country_metrics,
        x='train_rows',
        y='medae_real',
        hue='region',
        size='test_rows',
        sizes=(40, 250),
        alpha=0.8,
        ax=axes[1]
    )
    label_df = country_metrics.sort_values('medae_real', ascending=False).head(REPORT_COUNTRY_LABELS)
    for _, row in label_df.iterrows():
        axes[1].annotate(
            row['country_clean'],
            (row['train_rows'], row['medae_real']),
            fontsize=8,
            alpha=0.8
        )
    axes[1].set_xscale('log')
    axes[1].set_title('Country Error vs Training Support')
    axes[1].set_xlabel('Training rows by country (log scale)')
    axes[1].set_ylabel('Test MedAE in 2025 USD')
    format_money_axis(axes[1], axis='y')

    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# Compares predicted and actual compensation medians across the main geography cuts
def plot_geography_alignment(report_bundle, path):
    scored = report_bundle['selected_scored']

    region_view = (
        scored
        .groupby('region', as_index=False)
        .agg(
            actual_median_real=('actual_real', 'median'),
            pred_median_real=('pred_real', 'median')
        )
    )
    top_countries = (
        scored
        .groupby('country_clean')
        .size()
        .sort_values(ascending=False)
        .head(REPORT_COUNTRY_LABELS)
        .index
    )
    country_view = (
        scored.loc[scored['country_clean'].isin(top_countries)]
        .groupby('country_clean', as_index=False)
        .agg(
            rows=('actual_real', 'size'),
            actual_median_real=('actual_real', 'median'),
            pred_median_real=('pred_real', 'median')
        )
        .sort_values('rows', ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    region_plot = region_view.melt(
        id_vars='region',
        value_vars=['actual_median_real', 'pred_median_real'],
        var_name='series',
        value_name='comp_real'
    )
    sns.barplot(data=region_plot, x='comp_real', y='region', hue='series', ax=axes[0], palette='Set2')
    axes[0].set_title('Actual vs Predicted Region Median Compensation')
    axes[0].set_xlabel('Median compensation in 2025 USD')
    axes[0].set_ylabel('')
    format_money_axis(axes[0], axis='x')

    country_plot = country_view.melt(
        id_vars='country_clean',
        value_vars=['actual_median_real', 'pred_median_real'],
        var_name='series',
        value_name='comp_real'
    )
    sns.barplot(data=country_plot, x='comp_real', y='country_clean', hue='series', ax=axes[1], palette='Set2')
    axes[1].set_title('Actual vs Predicted Country Medians on Top Test Countries')
    axes[1].set_xlabel('Median compensation in 2025 USD')
    axes[1].set_ylabel('')
    format_money_axis(axes[1], axis='x')

    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# SHAP bar chart keeps the top drivers readable for a slide or report page
def plot_shap_bar(report_bundle, path, max_display=20):
    shap_bundle = report_bundle['shap_bundle']
    plt.figure(figsize=(12, 7))
    shap.summary_plot(
        shap_bundle['shap_values'],
        shap_bundle['display_frame'],
        plot_type='bar',
        max_display=max_display,
        show=False
    )
    plt.title('SHAP Summary: Top Compensation Drivers')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


# SHAP beeswarm shows both feature strength and direction for the selected model
def plot_shap_beeswarm(report_bundle, path, max_display=20):
    shap_bundle = report_bundle['shap_bundle']
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_bundle['shap_values'],
        shap_bundle['display_frame'],
        max_display=max_display,
        show=False
    )
    plt.title('SHAP Beeswarm: Selected Compensation Model')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


# Family level SHAP helps show the broader story hidden behind the many individual tech flags
def plot_shap_family_importance(report_bundle, path):
    shap_family = report_bundle['shap_bundle']['family_importance']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=shap_family, x='mean_abs_shap', y='family', ax=ax, palette='mako')
    ax.set_title('SHAP Importance By Feature Family')
    ax.set_xlabel('Total mean |SHAP|')
    ax.set_ylabel('')
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# Rolling origin figure checks whether the locked model stays competitive as history expands
def plot_rolling_origin(report_bundle, path):
    rolling = report_bundle['rolling']
    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)

    for row in rolling['results']:
        folds = row['folds'].copy()
        if folds.empty:
            continue
        folds = folds.sort_values('valid_year')
        axes[0].plot(folds['valid_year'], folds['medae_real'], marker='o', label=row['setup'])
        axes[1].plot(folds['valid_year'], folds['r2_log'], marker='o', label=row['setup'])

    axes[0].set_title('Rolling Origin Validation MedAE')
    axes[0].set_ylabel('MedAE in 2025 USD')
    format_money_axis(axes[0], axis='y')

    axes[1].set_title('Rolling Origin Validation R2 On Log Compensation')
    axes[1].set_xlabel('Validation year')
    axes[1].set_ylabel('R2 on log target')

    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()
    fig.legend(handles, labels, title='Setup', loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# Compares the two United States model views at the split and metric level
def plot_us_compare(report_bundle, path):
    summary = report_bundle['summary'].copy()
    split_counts = (
        summary[[
                'model_view',
                'train_rows',
                'valid_rows',
                'test_rows'
        ]].melt(
            id_vars='model_view',
            value_vars=['train_rows', 'valid_rows', 'test_rows'],
            var_name='split',
            value_name='rows'
        ).assign(split=lambda df: df['split'].str.replace('_rows', '', regex=False))
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.barplot(data=split_counts, x='split', y='rows', hue='model_view', ax=axes[0, 0], palette='Set2')
    axes[0, 0].set_title('Modeled Rows By Split')
    axes[0, 0].set_xlabel('Split')
    axes[0, 0].set_ylabel('Rows')
    axes[0, 0].legend(title='Model view')

    metric_plots = [
        ('valid_medae_real', 'Validation MedAE', axes[0, 1]),
        ('test_medae_real', 'Test MedAE', axes[1, 0]),
        ('test_rmse_real', 'Test RMSE', axes[1, 1])
    ]
    for metric_col, title, ax in metric_plots:
        sns.barplot(data=summary, x=metric_col, y='model_view', ax=ax, color='#72B7B2')
        ax.set_title(title)
        ax.set_xlabel('Compensation in 2025 USD')
        ax.set_ylabel('')
        format_money_axis(ax, axis='x')
        for container in ax.containers:
            labels = [money_formatter(value, None) for value in container.datavalues]
            ax.bar_label(container, labels=labels, padding=3, fontsize=9)

    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# Shows how the United States only refit behaves on the held out 2025 rows
def plot_us_diagnostics(report_bundle, path):
    country_scored = report_bundle['country_test_scored']
    global_scored = report_bundle['global_test_scored']
    decile_compare = report_bundle['decile_compare'].copy()
    diag_min = min(country_scored['actual_real'].min(), country_scored['pred_real'].min())
    diag_max = max(country_scored['actual_real'].max(), country_scored['pred_real'].max())
    diag_df = country_scored[['actual_real', 'pred_real']].dropna()
    if len(diag_df) > REPORT_SCATTER_SAMPLE:
        diag_df = diag_df.sample(REPORT_SCATTER_SAMPLE, random_state=RANDOM_STATE)

    error_compare = pd.concat([
        global_scored[['signed_error_real']].assign(model_view=GLOBAL_MAIN_VIEW),
        country_scored[['signed_error_real']].assign(model_view=US_REFIT_VIEW)
    ]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    axes[0].scatter(
        diag_df['actual_real'],
        diag_df['pred_real'],
        s=12,
        alpha=0.18,
        color='#4C78A8',
        edgecolors='none',
        rasterized=len(diag_df) > 2000
    )
    axes[0].plot([diag_min, diag_max], [diag_min, diag_max], color='#F58518', linestyle='--', linewidth=1.5)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlim(diag_min * 0.95, diag_max * 1.05)
    axes[0].set_ylim(diag_min * 0.95, diag_max * 1.05)
    axes[0].set_title('United States Refit: Actual vs Predicted')
    axes[0].set_xlabel('Actual compensation in 2025 USD')
    axes[0].set_ylabel('Predicted compensation in 2025 USD')
    format_money_axis(axes[0], axis='both')

    sns.histplot(
        data=error_compare,
        x='signed_error_real',
        hue='model_view',
        bins=35,
        element='step',
        stat='count',
        common_norm=False,
        ax=axes[1]
    )
    axes[1].axvline(0, color='black', linewidth=1, linestyle='--')
    axes[1].set_title('Signed Error Comparison')
    axes[1].set_xlabel('Prediction minus actual in 2025 USD')
    axes[1].set_ylabel('Rows')
    format_money_axis(axes[1], axis='x')

    sns.lineplot(
        data=decile_compare,
        x='actual_decile',
        y='medae_real',
        hue='setup',
        marker='o',
        ax=axes[2]
    )
    axes[2].set_title('Test MedAE By Compensation Decile')
    axes[2].set_xlabel('Actual compensation decile')
    axes[2].set_ylabel('Median absolute error in 2025 USD')
    format_money_axis(axes[2], axis='y')

    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# Compares feature importance shares before and after the country is held fixed
def plot_us_feature_shift(report_bundle, path):
    shap_compare = report_bundle['shap_compare'].copy()
    top_features = (
        shap_compare
        .groupby('feature')['normalized_mean_abs_shap']
        .max()
        .sort_values(ascending=False)
        .head(REPORT_US_TOP_FEATURES)
        .index
    )
    plot_df = shap_compare.loc[shap_compare['feature'].isin(top_features)].copy()
    feature_order = (
        plot_df
        .groupby('feature')['normalized_mean_abs_shap']
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    plot_df['feature'] = pd.Categorical(plot_df['feature'], categories=feature_order, ordered=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=plot_df,
        x='normalized_mean_abs_shap',
        y='feature',
        hue='model_view',
        ax=ax,
        palette='viridis'
    )
    ax.set_title('Global vs United States Feature Shift')
    ax.set_xlabel('Normalized mean |SHAP| share')
    ax.set_ylabel('')
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# SHAP beeswarm for the United States only refit keeps the local driver story visible
def plot_us_shap_beeswarm(report_bundle, path, max_display=20):
    shap_bundle = report_bundle['country_shap_bundle']
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_bundle['shap_values'],
        shap_bundle['display_frame'],
        max_display=max_display,
        show=False
    )
    plt.title('SHAP Beeswarm: United States only refit')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


# Writes the tables, figures, and manifest so the report has one reproducible source
def generate_compensation_report(
    clean_core,
    output_dir=REPORT_DIR,
    shap_sample_size=REPORT_SHAP_SAMPLE
):
    plt.switch_backend('Agg')
    report_bundle = build_compensation_report_bundle(
        clean_core,
        shap_sample_size=shap_sample_size,
        include_shap=True,
        include_rolling=True
    )
    paths = build_report_paths(output_dir)

    report_bundle['comparison']['summary'].to_csv(paths['tables']['setup_summary'], index=False)
    report_bundle['split_counts'].to_csv(paths['tables']['split_counts'], index=False)
    report_bundle['region_metrics'].to_csv(paths['tables']['region_metrics'], index=False)
    report_bundle['country_metrics'].to_csv(paths['tables']['country_metrics'], index=False)
    report_bundle['decile_compare'].to_csv(paths['tables']['decile_compare'], index=False)
    report_bundle['feature_importance'].to_csv(paths['tables']['feature_importance'], index=False)
    report_bundle['shap_bundle']['top_features'].to_csv(paths['tables']['shap_top_features'], index=False)
    report_bundle['shap_bundle']['family_importance'].to_csv(paths['tables']['shap_family'], index=False)

    plot_compensation_context(report_bundle, paths['figures']['context'])
    plot_setup_comparison(report_bundle, paths['figures']['setup_compare'])
    plot_winsor_vs_plain_by_decile(report_bundle, paths['figures']['winsor_vs_plain'])
    plot_test_diagnostics(report_bundle, paths['figures']['test_diag'])
    plot_subgroup_error(report_bundle, paths['figures']['subgroup_error'])
    plot_geography_alignment(report_bundle, paths['figures']['geo_alignment'])
    plot_shap_bar(report_bundle, paths['figures']['shap_bar'])
    plot_shap_beeswarm(report_bundle, paths['figures']['shap_beeswarm'])
    plot_shap_family_importance(report_bundle, paths['figures']['shap_family'])
    plot_rolling_origin(report_bundle, paths['figures']['rolling_origin'])

    manifest = {
        'objective': 'Report the locked compensation model with reproducible visuals and tables',
        'unit_of_analysis': 'respondent-year',
        'task_type': 'predictive compensation regression',
        'selected_setup': report_bundle['comparison']['selected_main_setup'],
        'train_years': CORE_WINDOW_YEARS,
        'valid_year': VALID_YEAR,
        'test_year': TEST_YEAR,
        'rows': {
            'train': int(len(report_bundle['core_train'])),
            'valid': int(len(report_bundle['core_valid'])),
            'test': int(len(report_bundle['core_test']))
        },
        'selected_metrics': {
            'valid_medae_real': float(report_bundle['selected_result']['valid_metrics']['medae_real']),
            'test_medae_real': float(report_bundle['selected_result']['test_metrics']['medae_real']),
            'test_rmse_real': float(report_bundle['selected_result']['test_metrics']['rmse_real']),
            'test_r2_log': float(report_bundle['selected_result']['test_metrics']['r2_log'])
        },
        'artifacts': {
            'figures': {name: str(path) for name, path in paths['figures'].items()},
            'tables': {name: str(path) for name, path in paths['tables'].items()}
        },
        'notes': [
            'Compensation is modeled in log 2025 USD and scored back in real dollars',
            'The selected model is the winsorized target LightGBM with train fit top tech flags',
            'Geography remains the dominant source of signal and the largest source of subgroup error differences',
            'The survey is repeated cross sectional and not a longitudinal panel'
        ]
    }
    paths['manifest'].write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    return {
        'report_bundle': report_bundle,
        'paths': paths,
        'manifest': manifest
    }


# Writes the United States side report artifacts for the locked compensation model
def generate_us_compensation_report(
    clean_core,
    output_dir=REPORT_US_DIR,
    country=REPORT_US_COUNTRY,
    shap_sample_size=REPORT_SHAP_SAMPLE
):
    plt.switch_backend('Agg')
    report_bundle = build_us_compensation_report_bundle(
        clean_core,
        country=country,
        shap_sample_size=shap_sample_size,
        include_shap=True
    )
    paths = build_us_report_paths(output_dir)

    report_bundle['summary'].to_csv(paths['tables']['summary'], index=False)
    report_bundle['predictions'].to_csv(paths['tables']['test_predictions'], index=False)
    report_bundle['shap_compare'].to_csv(paths['tables']['shap_compare'], index=False)
    report_bundle['shap_top_features'].to_csv(paths['tables']['shap_top_features'], index=False)

    plot_us_compare(report_bundle, paths['figures']['compare'])
    plot_us_diagnostics(report_bundle, paths['figures']['diagnostics'])
    plot_us_feature_shift(report_bundle, paths['figures']['feature_shift'])
    plot_us_shap_beeswarm(report_bundle, paths['figures']['shap_beeswarm'])

    global_row = report_bundle['summary'].loc[report_bundle['summary']['model_view'].eq(GLOBAL_MAIN_VIEW)].iloc[0]
    country_row = report_bundle['summary'].loc[report_bundle['summary']['model_view'].eq(US_REFIT_VIEW)].iloc[0]
    manifest = {
        'objective': 'Report the locked compensation model with a United States only side analysis',
        'unit_of_analysis': 'respondent-year',
        'task_type': 'predictive compensation regression',
        'country': country,
        'train_years': CORE_WINDOW_YEARS,
        'valid_year': VALID_YEAR,
        'test_year': TEST_YEAR,
        'rows': {
            'train': int(len(report_bundle['country_train'])),
            'valid': int(len(report_bundle['country_valid'])),
            'test': int(len(report_bundle['country_test']))
        },
        'metrics': {
            'global_main_model': {
                'valid_medae_real': float(global_row['valid_medae_real']),
                'test_medae_real': float(global_row['test_medae_real']),
                'test_rmse_real': float(global_row['test_rmse_real']),
                'test_r2_log': float(global_row['test_r2_log'])
            },
            'us_only_refit': {
                'valid_medae_real': float(country_row['valid_medae_real']),
                'test_medae_real': float(country_row['test_medae_real']),
                'test_rmse_real': float(country_row['test_rmse_real']),
                'test_r2_log': float(country_row['test_r2_log'])
            }
        },
        'artifacts': {
            'figures': {name: str(path) for name, path in paths['figures'].items()},
            'tables': {name: str(path) for name, path in paths['tables'].items()}
        },
        'notes': [
            'The compare figure, summary table, and diagnostic comparisons use the global main model on its full contract against the United States only refit',
            'The prediction table now writes scored rows for the full global 2025 test set and the United States only 2025 refit set in one long-form export',
            'The SHAP comparison now contrasts the full global main model against the United States only refit rather than a global-on-United-States slice',
            'The side refit removes country and region because the country is held fixed',
            'The United States only refit keeps the same winsorized target LightGBM family and top tech flag logic',
            'This side report is explanatory and does not replace the global compensation model'
        ]
    }
    paths['manifest'].write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    return {
        'report_bundle': report_bundle,
        'paths': paths,
        'manifest': manifest
    }


# -------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------

# CLI args keep the module runnable for either quick summaries or full reporting output
def parse_args():
    parser = argparse.ArgumentParser(description='Run the canonical compensation model workflow')
    parser.add_argument('--input-path', default=str(CLEAN_PATH), help='Path to clean_core parquet')
    report_group = parser.add_mutually_exclusive_group()
    report_group.add_argument('--report', action='store_true', help='Write the final compensation reporting visuals and tables')
    report_group.add_argument('--report-us', action='store_true', help='Write the United States compensation side report artifacts')
    parser.add_argument('--output-dir', default=None, help='Directory for compensation reporting artifacts')
    parser.add_argument('--shap-sample-size', type=int, default=REPORT_SHAP_SAMPLE, help='Max rows used for SHAP reporting plots')
    return parser.parse_args()


# Runs the canonical compensation comparison table directly from the cleaned parquet
def main():
    args = parse_args()
    clean_core = comp_clean.load_clean_core(Path(args.input_path))

    if args.report:
        report = generate_compensation_report(
            clean_core,
            output_dir=Path(args.output_dir) if args.output_dir is not None else REPORT_DIR,
            shap_sample_size=args.shap_sample_size
        )
        print(f"Report directory: {report['paths']['output_dir']}")
        for name, path in report['paths']['figures'].items():
            print(f"figure[{name}]: {path}")
        for name, path in report['paths']['tables'].items():
            print(f"table[{name}]: {path}")
        print(f"manifest: {report['paths']['manifest']}")
        return

    if args.report_us:
        report = generate_us_compensation_report(
            clean_core,
            output_dir=Path(args.output_dir) if args.output_dir is not None else REPORT_US_DIR,
            country=REPORT_US_COUNTRY,
            shap_sample_size=args.shap_sample_size
        )
        print(f"US report directory: {report['paths']['output_dir']}")
        for name, path in report['paths']['figures'].items():
            print(f"figure[{name}]: {path}")
        for name, path in report['paths']['tables'].items():
            print(f"table[{name}]: {path}")
        print(f"manifest: {report['paths']['manifest']}")
        return

    results = compare_same_sample_setups(clean_core)

    print(results['summary'].to_string(index=False))
    print(f"Selected main setup: {results['selected_main_setup']}")


if __name__ == '__main__':
    main()
