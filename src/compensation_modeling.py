import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import model_audit

RANDOM_STATE = 42
TARGET_COL = model_audit.COMP_TARGET_LOG
REAL_TARGET_COL = model_audit.COMP_TARGET_REAL
REAL_WINSOR_COL = 'comp_real_2025_winsor'
WINSOR_TARGET_COL = 'log_comp_real_2025_winsor'
CORE_WINDOW_YEARS = [2019, 2020, 2021, 2022, 2023]
TECH_WINDOW_YEARS = [2021, 2022, 2023]
AI_WINDOW_YEARS = [2023]
VALID_YEAR = 2024
TEST_YEAR = 2025
TOP_N_TECH = {
    'language': 15,
    'database': 10,
    'platform': 10
}
BASELINE_GROUP_SETS = [
    ['country_clean'],
    ['region'],
    []
]
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
NUMERIC_FIELDS = [
    'survey_year',
    'age_mid',
    'years_code_pro_clean',
    'work_exp_clean',
    'professional_experience_years',
    'language_count',
    'database_count',
    'platform_count',
    'webframe_count',
    'misc_tech_count',
    'learn_code_count',
    'learn_code_online_count',
    'coding_activities_count',
    'op_sys_prof_count',
    'is_full_time_employed',
    'is_part_time_employed',
    'is_independent',
    'is_student_status',
    'is_retired_status',
    'comp_usd_clean',
    TARGET_COL,
    REAL_TARGET_COL,
    WINSOR_TARGET_COL,
    REAL_WINSOR_COL
]


# -------------------------------------------------------------------------------------
# Frame prep
# -------------------------------------------------------------------------------------
def coerce_numeric_fields(frame, fields=None):
    out = frame.copy()
    fields = NUMERIC_FIELDS if fields is None else fields

    for field in fields:
        if field in out.columns:
            out[field] = pd.to_numeric(out[field], errors='coerce')

    role_cols = [col for col in out.columns if col.startswith('role_')]
    for field in role_cols:
        out[field] = pd.to_numeric(out[field], errors='coerce')

    return out


def add_survey_year_str(frame):
    out = frame.copy()
    out['survey_year_str'] = out['survey_year'].astype('string')
    return out


def coerce_compensation_frame(clean_core):
    frame = add_survey_year_str(clean_core)
    numeric_fields = NUMERIC_FIELDS + [col for col in frame.columns if col.startswith('role_')]
    frame = coerce_numeric_fields(frame, numeric_fields)

    if 'survey_year_str' in frame.columns:
        frame['survey_year_str'] = frame['survey_year_str'].astype(object)

    text_cols = frame.select_dtypes(include=['object', 'string']).columns
    for col in text_cols:
        frame.loc[frame[col].isna(), col] = np.nan

    return frame


def get_comp_frames(clean_core):
    base = coerce_compensation_frame(clean_core)
    base = base.loc[
        base['is_comp_model_sample']
        & base['country_clean'].notna()
        & base['region'].notna()
    ].copy()

    core_df = base.copy()
    tech_df = base.loc[base['survey_year'].ge(min(TECH_WINDOW_YEARS))].copy()
    ai_df = base.loc[base['survey_year'].ge(min(AI_WINDOW_YEARS))].copy()
    return core_df, tech_df, ai_df


def prepare_compensation_frame(clean_core):
    core_df, _, _ = get_comp_frames(clean_core)
    return core_df.copy()


def get_role_cols(frame):
    return sorted(col for col in frame.columns if col.startswith('role_'))


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


def split_window_with_tech_flags(frame, train_years, valid_year, test_year, top_n_map=None):
    train_df, valid_df, test_df = model_audit.split_years(frame, train_years, valid_year, test_year)
    train_df, tech_flag_cols = add_top_tech_flags(train_df, top_n_map=top_n_map, fit_frame=train_df)
    valid_df, _ = add_top_tech_flags(valid_df, top_n_map=top_n_map, fit_frame=train_df)
    test_df, _ = add_top_tech_flags(test_df, top_n_map=top_n_map, fit_frame=train_df)
    return train_df, valid_df, test_df, tech_flag_cols


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
def normalize_feature_frame(frame, cat_cols, num_cols):
    out = frame.copy()

    for col in cat_cols:
        out[col] = out[col].astype(object)
        out.loc[out[col].isna(), col] = np.nan

    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors='coerce').astype(float)

    return out


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


def score_holdout_baseline(train_df, valid_df, test_df, group_sets=None):
    group_sets = BASELINE_GROUP_SETS if group_sets is None else group_sets
    valid_metrics = model_audit.score_hier_median(train_df, valid_df, group_sets)
    train_valid = pd.concat([train_df, valid_df], axis=0)
    test_metrics = model_audit.score_hier_median(train_valid, test_df, group_sets)
    return {
        'valid_metrics': valid_metrics,
        'test_metrics': test_metrics
    }


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
    valid_metrics = model_audit.score_log_comp(
        valid_prepped[target_col],
        early_model.predict(valid_prepped[features], num_iteration=best_iteration)
    )

    final_params = dict(resolved_params)
    final_params['n_estimators'] = best_iteration
    final_model = lgb.LGBMRegressor(**final_params)
    full_train_raw = pd.concat([train_df, valid_df], axis=0)
    full_train, full_num_imputer = prepare_lgbm_frame(full_train_raw, cat_cols, num_cols, target_col=target_col)
    full_test = transform_lgbm_frame(test_df, cat_cols, num_cols, full_num_imputer, target_col=target_col)
    final_model.fit(full_train[features], full_train[target_col].to_numpy())
    test_metrics = model_audit.score_log_comp(
        full_test[target_col],
        final_model.predict(full_test[features])
    )

    return {
        'model': final_model,
        'best_params': final_params,
        'valid_metrics': valid_metrics,
        'test_metrics': test_metrics,
        'feature_cols': features,
        'num_imputer': full_num_imputer
    }


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
        'num_imputer': full_num_imputer
    }


# -------------------------------------------------------------------------------------
# Canonical workflows
# -------------------------------------------------------------------------------------
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
