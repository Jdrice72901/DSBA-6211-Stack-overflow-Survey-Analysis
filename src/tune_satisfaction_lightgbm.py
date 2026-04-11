import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

import src.satisfaction_modeling as sat
from src import model_audit

# -------------------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------------------

# Canonical clean input plus the output root for tuning artifacts
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / 'data' / 'derived' / 'clean_core.parquet'
OUTPUT_ROOT = ROOT / 'data' / 'outputs'

# Default study names and run lengths for the standalone Optuna workflow
DEF_STUDY = 'sat_lightgbm_optuna'
DEF_TRIALS = 80
DEF_TIMEOUT = None

# Seed presets for the tuning objective, final rescoring, and optional rolling-origin branch
DEF_SEED_TEXT = '42,52,62'
DEF_FINAL_SEED_TEXT = '42,52,62,72,82'
DEF_ROLLING_SEED_TEXT = '42'

# Default objective and search-space controls for the LightGBM study
DEF_OBJECTIVE_METRIC = 'qwk'
DEF_SEARCH_SPACE = 'broad'
DEF_EARLY_STOPPING = 75
DEF_GPU_BACKEND = 'gpu'

# Extra anchor values for knobs that are not explicitly set in the canonical default params
ANCHOR_OVERRIDES = {
    'min_child_weight': 1e-3,
    'subsample_freq': 0,
    'min_split_gain': 0.0,
    'cat_l2': 10.0,
    'cat_smooth': 10.0,
    'max_cat_to_onehot': 4,
    'max_bin': 255
}


# -------------------------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------------------------

# Keeps long cloud runs chatty enough that you know they are still alive
def log_line(message=''):
    print(message, flush=True)


# Parses comma-separated seed strings into a clean list of ints
def parse_seed_text(seed_text, default_text):
    text = default_text if seed_text is None else str(seed_text).strip()
    if not text:
        raise ValueError('Seed list cannot be empty')

    seeds = [int(part.strip()) for part in text.split(',') if part.strip()]
    if not seeds:
        raise ValueError('Seed list could not be parsed')
    return seeds


# Reads the canonical cleaned table without caring whether it is parquet or csv
def read_frame(path):
    suffix = path.suffix.lower()
    if suffix == '.parquet':
        return pd.read_parquet(path)
    if suffix == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Unsupported input format: {path}')


# Summarizes repeated-seed runs so Optuna compares stable aggregates, not one lucky seed
def summarize_runs(run_df, prefix):
    return {
        f'{prefix}_qwk_mean': float(run_df['qwk'].mean()),
        f'{prefix}_qwk_std': float(run_df['qwk'].std(ddof=0)),
        f'{prefix}_macro_f1_mean': float(run_df['macro_f1'].mean()),
        f'{prefix}_macro_f1_std': float(run_df['macro_f1'].std(ddof=0)),
        f'{prefix}_accuracy_mean': float(run_df['accuracy'].mean()),
        f'{prefix}_accuracy_std': float(run_df['accuracy'].std(ddof=0)),
        f'{prefix}_weighted_f1_mean': float(run_df['weighted_f1'].mean()),
        f'{prefix}_adjacent_error_rate_mean': float(run_df['adjacent_error_rate'].mean()),
        f'{prefix}_far_miss_rate_mean': float(run_df['far_miss_rate'].mean()),
        f'{prefix}_best_iteration_mean': float(run_df['best_iteration'].mean()),
        f'{prefix}_best_iteration_median': int(run_df['best_iteration'].median())
    }


# Lets the study optimize QWK, macro F1, or a simple blend without changing the trial code
def objective_from_summary(summary, prefix, metric_name):
    if metric_name == 'macro_f1':
        return float(summary[f'{prefix}_macro_f1_mean'])
    if metric_name == 'qwk_macro_blend':
        qwk = float(summary[f'{prefix}_qwk_mean'])
        macro_f1 = float(summary[f'{prefix}_macro_f1_mean'])
        return 0.75 * qwk + 0.25 * macro_f1
    return float(summary[f'{prefix}_qwk_mean'])


# Centralizes the file paths written by the tuning run
def build_output_paths(study_name, output_root):
    study_dir = output_root / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    return {
        'study_dir': study_dir,
        'storage_path': study_dir / f'{study_name}.sqlite3',
        'trials_path': study_dir / f'{study_name}_trials.csv',
        'summary_path': study_dir / f'{study_name}_summary.csv',
        'manifest_path': study_dir / f'{study_name}_manifest.json',
        'best_params_path': study_dir / f'{study_name}_best_params.json',
        'test_runs_path': study_dir / f'{study_name}_test_runs.csv'
    }


# Keeps persistence opt-in so sqlite is only created when resumability is worth the overhead
def resolve_storage(args, output_paths):
    if args.storage_path is not None:
        storage_path = Path(args.storage_path)
        return {
            'persist_study': True,
            'storage_path': storage_path,
            'storage_uri': f'sqlite:///{storage_path}'
        }

    if args.resume and not args.persist_study:
        raise ValueError('--resume requires --persist-study or an explicit --storage-path')

    if args.persist_study:
        storage_path = output_paths['storage_path']
        return {
            'persist_study': True,
            'storage_path': storage_path,
            'storage_uri': f'sqlite:///{storage_path}'
        }

    return {
        'persist_study': False,
        'storage_path': None,
        'storage_uri': None
    }


# -------------------------------------------------------------------------------------
# Data prep
# -------------------------------------------------------------------------------------

# Builds the exact canonical satisfaction frame and split bundle used by the main pipeline
def build_run_bundle(args):
    clean_core = read_frame(Path(args.input_path))
    bundle = sat.build_satisfaction_bundle(
        clean_core,
        numeric_scheme=args.numeric_scheme,
        drop_2018=args.drop_2018
    )
    frame = bundle['frame']
    feature_sets = bundle['feature_sets']

    if args.spec == 'core_with_comp':
        frame = frame.loc[frame['log_comp_real_2025'].notna()].copy()

    cat_cols = feature_sets[f'{args.spec}_cat']
    num_cols = feature_sets[f'{args.spec}_num']
    train_df, valid_df, test_df = sat.split_satisfaction_years(frame)

    if not len(train_df) or not len(valid_df) or not len(test_df):
        raise ValueError('One of the canonical splits is empty')

    holdout_bundle = build_dataset_bundle(train_df, valid_df, cat_cols, num_cols)
    rolling_bundles = []
    if float(args.rolling_weight) > 0:
        for train_years, valid_year, fold_train, fold_valid in model_audit.rolling_origin_splits(
            frame,
            min_train_year=args.rolling_min_train_year,
            final_valid_year=args.rolling_final_valid_year
        ):
            rolling_bundles.append({
                'train_years': train_years,
                'valid_year': valid_year,
                'bundle': build_dataset_bundle(fold_train, fold_valid, cat_cols, num_cols)
            })

    train_valid = pd.concat([train_df, valid_df], axis=0)
    final_train, final_num_prep = sat.prepare_lgbm_frame(train_valid, cat_cols, num_cols)
    final_test = sat.transform_lgbm_frame(test_df, cat_cols, num_cols, final_num_prep)

    return {
        'clean_core': clean_core,
        'frame': frame,
        'train_df': train_df,
        'valid_df': valid_df,
        'test_df': test_df,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'features': cat_cols + num_cols,
        'holdout_bundle': holdout_bundle,
        'rolling_bundles': rolling_bundles,
        'final_train': final_train,
        'final_test': final_test,
        'feature_sets': feature_sets
    }


# Preps one train-valid bundle so repeated seed fits don't redo the same frame work
def build_dataset_bundle(train_df, valid_df, cat_cols, num_cols):
    train_prepped, num_prep = sat.prepare_lgbm_frame(train_df, cat_cols, num_cols)
    valid_prepped = sat.transform_lgbm_frame(valid_df, cat_cols, num_cols, num_prep)
    return {
        'train': train_prepped,
        'valid': valid_prepped,
        'features': cat_cols + num_cols
    }


# -------------------------------------------------------------------------------------
# Search setup
# -------------------------------------------------------------------------------------

# These are the LightGBM knobs we actually want Optuna to explore
def tune_param_keys():
    return [
        'n_estimators',
        'learning_rate',
        'num_leaves',
        'max_depth',
        'min_child_samples',
        'min_child_weight',
        'subsample',
        'subsample_freq',
        'colsample_bytree',
        'reg_alpha',
        'reg_lambda',
        'min_split_gain',
        'cat_l2',
        'cat_smooth',
        'max_cat_to_onehot',
        'max_bin'
    ]


# Uses the current canonical LightGBM setup as the first anchor trial
def default_anchor_params():
    anchor = sat.resolve_lgb_params()
    anchor.update(ANCHOR_OVERRIDES)
    return {key: anchor[key] for key in tune_param_keys()}


# Collects any optional GPU arguments into one clean params dict
def resolve_device_params(args):
    params = {}
    if args.use_gpu:
        params['device_type'] = args.gpu_backend
        params['gpu_device_id'] = int(args.gpu_device_id)
        if args.gpu_backend == 'gpu' and args.gpu_platform_id is not None:
            params['gpu_platform_id'] = int(args.gpu_platform_id)
        if args.gpu_backend == 'gpu' and args.gpu_use_dp:
            params['gpu_use_dp'] = True
    return params


# Broad and focused search spaces so local smoke runs and cloud studies can share one script
def suggest_lgbm_params(trial, args):
    if args.search_space == 'focused':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 250, 1000, step=25),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 191),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 6, 8, 10, 12]),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 120),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 5.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.65, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 3),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.25),
            'cat_l2': trial.suggest_float('cat_l2', 1.0, 25.0, log=True),
            'cat_smooth': trial.suggest_float('cat_smooth', 1.0, 50.0, log=True),
            'max_cat_to_onehot': trial.suggest_int('max_cat_to_onehot', 4, 16),
            'max_bin': trial.suggest_categorical('max_bin', [127, 255])
        }
    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 2500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.12, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 4, 6, 8, 10, 12, 14, 16]),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 20.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 7),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 25.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 25.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'cat_l2': trial.suggest_float('cat_l2', 1.0, 50.0, log=True),
            'cat_smooth': trial.suggest_float('cat_smooth', 1.0, 200.0, log=True),
            'max_cat_to_onehot': trial.suggest_int('max_cat_to_onehot', 4, 32),
            'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255])
        }

    resolved = sat.resolve_lgb_params()
    resolved.update(resolve_device_params(args))
    resolved.update(params)
    return resolved


# Fans a base param set out across explicit seeds for more stable comparisons
def seed_params(params, seed):
    resolved = dict(params)
    resolved['random_state'] = int(seed)
    resolved['bagging_seed'] = int(seed)
    resolved['feature_fraction_seed'] = int(seed)
    resolved['data_random_seed'] = int(seed)
    return resolved


# -------------------------------------------------------------------------------------
# Model fitting
# -------------------------------------------------------------------------------------

# Fits one valid fold and returns the full metric bundle plus best iteration
def fit_valid_once(bundle, params, seed, early_stopping_rounds):
    seeded = seed_params(params, seed)
    model = lgb.LGBMClassifier(**seeded)
    model.fit(
        bundle['train'][bundle['features']],
        bundle['train'][sat.SAT_TARGET_COL].to_numpy(),
        eval_set=[(bundle['valid'][bundle['features']], bundle['valid'][sat.SAT_TARGET_COL].to_numpy())],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
    )
    best_iteration = model.best_iteration_ or seeded['n_estimators']
    pred = pd.Series(
        model.predict(bundle['valid'][bundle['features']], num_iteration=best_iteration),
        index=bundle['valid'].index
    ).astype(int)
    metrics = sat.score_satisfaction(bundle['valid'][sat.SAT_TARGET_COL], pred)
    metrics['best_iteration'] = int(best_iteration)
    return metrics


# Scores the main 2024 validation bundle across seeds, with Optuna pruning hooks if needed
def evaluate_bundle(bundle, params, seed_list, early_stopping_rounds, trial=None, step_offset=0, label='holdout'):
    rows = []
    for step_idx, seed in enumerate(seed_list, start=1):
        metrics = fit_valid_once(bundle, params, seed, early_stopping_rounds)
        rows.append({'seed': seed, **metrics})
        partial = pd.DataFrame(rows)
        partial_qwk = float(partial['qwk'].mean())
        if trial is not None:
            trial.report(partial_qwk, step=step_offset + step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned(f'Pruned during {label} evaluation')
    return pd.DataFrame(rows)


# Optional rolling-origin side score so tuning can penalize temporally brittle params
def evaluate_rolling_bundles(rolling_bundles, params, seed_list, early_stopping_rounds, trial=None, step_offset=0):
    rows = []
    step = step_offset
    for fold in rolling_bundles:
        for seed in seed_list:
            metrics = fit_valid_once(fold['bundle'], params, seed, early_stopping_rounds)
            rows.append({
                'train_years': ','.join(map(str, fold['train_years'])),
                'valid_year': int(fold['valid_year']),
                'seed': int(seed),
                **metrics
            })
            step += 1
            if trial is not None:
                partial = pd.DataFrame(rows)
                partial_qwk = float(partial['qwk'].mean())
                trial.report(partial_qwk, step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned('Pruned during rolling-origin evaluation')
    return pd.DataFrame(rows)


# Final post-selection test scoring so the default and tuned setups can be compared cleanly
def evaluate_final_on_test(run_bundle, params, seed_list, early_stopping_rounds, setup_name):
    rows = []
    features = run_bundle['features']

    for seed in seed_list:
        valid_metrics = fit_valid_once(run_bundle['holdout_bundle'], params, seed, early_stopping_rounds)
        seeded = seed_params(params, seed)
        seeded['n_estimators'] = int(valid_metrics['best_iteration'])
        final_model = lgb.LGBMClassifier(**seeded)
        final_model.fit(
            run_bundle['final_train'][features],
            run_bundle['final_train'][sat.SAT_TARGET_COL].to_numpy()
        )
        test_pred = pd.Series(
            final_model.predict(run_bundle['final_test'][features]),
            index=run_bundle['final_test'].index
        ).astype(int)
        test_metrics = sat.score_satisfaction(run_bundle['final_test'][sat.SAT_TARGET_COL], test_pred)
        rows.append({
            'setup': setup_name,
            'seed': int(seed),
            'valid_qwk': float(valid_metrics['qwk']),
            'valid_macro_f1': float(valid_metrics['macro_f1']),
            'valid_accuracy': float(valid_metrics['accuracy']),
            'best_iteration': int(valid_metrics['best_iteration']),
            'test_qwk': float(test_metrics['qwk']),
            'test_macro_f1': float(test_metrics['macro_f1']),
            'test_accuracy': float(test_metrics['accuracy']),
            'test_weighted_f1': float(test_metrics['weighted_f1']),
            'test_adjacent_error_rate': float(test_metrics['adjacent_error_rate']),
            'test_far_miss_rate': float(test_metrics['far_miss_rate'])
        })

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------------
# Optuna study
# -------------------------------------------------------------------------------------

# Builds the Optuna objective around the same year-aware contract as the canonical model
def build_objective(run_bundle, args):
    holdout_weight = float(args.holdout_weight)
    rolling_weight = float(args.rolling_weight)
    tune_seed_list = parse_seed_text(args.seed_list, DEF_SEED_TEXT)
    rolling_seed_list = parse_seed_text(args.rolling_seed_list, DEF_ROLLING_SEED_TEXT)

    def objective(trial):
        params = suggest_lgbm_params(trial, args)
        holdout_runs = evaluate_bundle(
            run_bundle['holdout_bundle'],
            params,
            tune_seed_list,
            args.early_stopping_rounds,
            trial=trial,
            step_offset=0,
            label='holdout'
        )
        holdout_summary = summarize_runs(holdout_runs, 'holdout')
        holdout_score = objective_from_summary(holdout_summary, 'holdout', args.objective_metric)

        rolling_summary = {}
        rolling_score = np.nan
        if rolling_weight > 0 and run_bundle['rolling_bundles']:
            rolling_runs = evaluate_rolling_bundles(
                run_bundle['rolling_bundles'],
                params,
                rolling_seed_list,
                args.early_stopping_rounds,
                trial=trial,
                step_offset=len(tune_seed_list)
            )
            rolling_summary = summarize_runs(rolling_runs, 'rolling')
            rolling_score = objective_from_summary(rolling_summary, 'rolling', args.objective_metric)

        total_weight = holdout_weight + rolling_weight
        if total_weight <= 0:
            raise ValueError('holdout_weight + rolling_weight must be positive')

        if rolling_weight > 0 and run_bundle['rolling_bundles']:
            objective_score = (
                holdout_weight * holdout_score + rolling_weight * rolling_score
            ) / total_weight
        else:
            objective_score = holdout_score

        for key, value in holdout_summary.items():
            trial.set_user_attr(key, value)
        for key, value in rolling_summary.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr('objective_metric', args.objective_metric)
        trial.set_user_attr('objective_score', float(objective_score))

        return float(objective_score)

    return objective


# Turns the Optuna study into a readable trials table for later reporting
def build_trials_frame(study):
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        rows.append({
            'trial': int(trial.number),
            'objective_score': float(trial.value),
            'holdout_qwk_mean': trial.user_attrs.get('holdout_qwk_mean'),
            'holdout_qwk_std': trial.user_attrs.get('holdout_qwk_std'),
            'holdout_macro_f1_mean': trial.user_attrs.get('holdout_macro_f1_mean'),
            'holdout_macro_f1_std': trial.user_attrs.get('holdout_macro_f1_std'),
            'holdout_accuracy_mean': trial.user_attrs.get('holdout_accuracy_mean'),
            'holdout_best_iteration_mean': trial.user_attrs.get('holdout_best_iteration_mean'),
            'holdout_best_iteration_median': trial.user_attrs.get('holdout_best_iteration_median'),
            'rolling_qwk_mean': trial.user_attrs.get('rolling_qwk_mean'),
            'rolling_macro_f1_mean': trial.user_attrs.get('rolling_macro_f1_mean'),
            'rolling_accuracy_mean': trial.user_attrs.get('rolling_accuracy_mean'),
            **trial.params
        })

    if not rows:
        raise ValueError('No completed Optuna trials were produced')

    return pd.DataFrame(rows).sort_values(
        ['objective_score', 'holdout_qwk_mean', 'holdout_macro_f1_mean'],
        ascending=False
    ).reset_index(drop=True)


# Tiny JSON writer so the best params and manifest stay human-readable
def write_json(data, path):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


# Converts numpy and Path objects into plain JSON-safe values
def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


# -------------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------------

# CLI arguments are explicit because this script is meant for long cloud runs, not notebook magic
def parse_args():
    parser = argparse.ArgumentParser(
        description='Robust Optuna tuner for the canonical LightGBM satisfaction model'
    )
    parser.add_argument('--input-path', default=str(INPUT_PATH), help='Path to clean_core parquet or csv')
    parser.add_argument('--study-name', default=DEF_STUDY, help='Optuna study name and artifact stem')
    parser.add_argument('--output-root', default=str(OUTPUT_ROOT), help='Directory for Optuna artifacts')
    parser.add_argument('--storage-path', default=None, help='Optional sqlite path for Optuna persistence')
    parser.add_argument('--persist-study', action='store_true', help='Persist the Optuna study to sqlite so long runs can resume')
    parser.add_argument('--spec', choices=['core_no_comp', 'core_with_comp'], default='core_no_comp')
    parser.add_argument('--numeric-scheme', choices=list(sat.SAT_NUMERIC_SCHEMES), default='default')
    parser.add_argument('--drop-2018', action='store_true', help='Drop the 2018 wave from the tuning frame')
    parser.add_argument('--n-trials', type=int, default=DEF_TRIALS, help='Number of Optuna trials to run')
    parser.add_argument('--timeout', type=int, default=DEF_TIMEOUT, help='Optional Optuna timeout in seconds')
    parser.add_argument('--random-seed', type=int, default=42, help='TPE sampler seed')
    parser.add_argument('--seed-list', default=DEF_SEED_TEXT, help='Comma-separated seeds scored in the main objective')
    parser.add_argument('--final-seed-list', default=DEF_FINAL_SEED_TEXT, help='Comma-separated seeds used for final default vs best evaluation')
    parser.add_argument('--rolling-seed-list', default=DEF_ROLLING_SEED_TEXT, help='Comma-separated seeds used for optional rolling-origin scoring')
    parser.add_argument('--holdout-weight', type=float, default=1.0, help='Weight on the 2024 validation objective')
    parser.add_argument('--rolling-weight', type=float, default=0.0, help='Weight on rolling-origin validation objective')
    parser.add_argument('--rolling-min-train-year', type=int, default=2015, help='First year allowed in rolling-origin training')
    parser.add_argument('--rolling-final-valid-year', type=int, default=2024, help='Last rolling-origin validation year')
    parser.add_argument('--objective-metric', choices=['qwk', 'macro_f1', 'qwk_macro_blend'], default=DEF_OBJECTIVE_METRIC)
    parser.add_argument('--search-space', choices=['focused', 'broad'], default=DEF_SEARCH_SPACE)
    parser.add_argument('--early-stopping-rounds', type=int, default=DEF_EARLY_STOPPING)
    parser.add_argument('--use-gpu', action='store_true', help='Use LightGBM GPU training')
    parser.add_argument('--gpu-backend', choices=['gpu', 'cuda'], default=DEF_GPU_BACKEND, help='LightGBM GPU backend: OpenCL gpu for wheel installs or cuda for CUDA builds')
    parser.add_argument('--gpu-device-id', type=int, default=0, help='GPU device id passed to LightGBM')
    parser.add_argument('--gpu-platform-id', type=int, default=None, help='Optional OpenCL platform id for the LightGBM gpu backend')
    parser.add_argument('--gpu-use-dp', action='store_true', help='Enable double precision on the OpenCL gpu backend if supported')
    parser.add_argument('--resume', action='store_true', help='Resume an existing persisted study if present')
    parser.add_argument('--quick', action='store_true', help='Lighter local preset with fewer trials and seeds')
    return parser.parse_args()


# Quick mode trims the study down for local smoke tests without touching the default cloud config
def apply_quick_preset(args):
    if not args.quick:
        return args

    if int(args.n_trials) == DEF_TRIALS:
        args.n_trials = 8
    if args.seed_list == DEF_SEED_TEXT:
        args.seed_list = '42'
    if args.final_seed_list == DEF_FINAL_SEED_TEXT:
        args.final_seed_list = '42'
    if args.search_space == DEF_SEARCH_SPACE:
        args.search_space = 'focused'
    if float(args.rolling_weight) == 0.0:
        args.rolling_weight = 0.0
    return args


# Runs the full tuning workflow and writes the readable artifacts to data/outputs
def main():
    args = apply_quick_preset(parse_args())
    output_paths = build_output_paths(args.study_name, Path(args.output_root))
    storage_cfg = resolve_storage(args, output_paths)
    storage_uri = storage_cfg['storage_uri']

    run_bundle = build_run_bundle(args)
    feature_count = len(run_bundle['features'])
    tune_seed_list = parse_seed_text(args.seed_list, DEF_SEED_TEXT)
    final_seed_list = parse_seed_text(args.final_seed_list, DEF_FINAL_SEED_TEXT)
    rolling_seed_list = parse_seed_text(args.rolling_seed_list, DEF_ROLLING_SEED_TEXT)

    log_line('[setup] Satisfaction LightGBM Optuna study')
    log_line(f'[setup] input={args.input_path}')
    log_line(f'[setup] study={args.study_name}')
    log_line(f'[setup] spec={args.spec}')
    log_line(
        f'[setup] rows train={len(run_bundle["train_df"]):,} '
        f'valid_2024={len(run_bundle["valid_df"]):,} '
        f'test_2025={len(run_bundle["test_df"]):,}'
    )
    log_line(f'[setup] features={feature_count}')
    log_line(
        f'[setup] objective_metric={args.objective_metric} '
        f'holdout_weight={float(args.holdout_weight):.2f} '
        f'rolling_weight={float(args.rolling_weight):.2f}'
    )
    log_line(f'[setup] tuning seeds={",".join(map(str, tune_seed_list))}')
    if float(args.rolling_weight) > 0:
        log_line(
            f'[setup] rolling folds={len(run_bundle["rolling_bundles"])} '
            f'rolling seeds={",".join(map(str, rolling_seed_list))}'
        )
    log_line(
        f'[setup] device={"gpu" if args.use_gpu else "cpu"} '
        f'gpu_backend={args.gpu_backend if args.use_gpu else "n/a"} '
        f'gpu_device_id={args.gpu_device_id if args.use_gpu else "n/a"}'
    )
    log_line(f'[setup] storage={storage_uri if storage_uri else "in-memory only"}')
    log_line('[setup] 2025 test remains untouched during Optuna search')

    study_kwargs = {
        'study_name': args.study_name,
        'direction': 'maximize',
        'sampler': optuna.samplers.TPESampler(seed=args.random_seed),
        'pruner': optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=max(2, len(tune_seed_list)))
    }
    if storage_uri:
        study_kwargs['storage'] = storage_uri
        study_kwargs['load_if_exists'] = args.resume

    study = optuna.create_study(**study_kwargs)

    if len(study.trials) == 0:
        study.enqueue_trial(default_anchor_params())

    objective = build_objective(run_bundle, args)
    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=args.timeout,
        show_progress_bar=True
    )

    trials_df = build_trials_frame(study)
    best_params = sat.resolve_lgb_params()
    best_params.update(resolve_device_params(args))
    best_params.update(study.best_trial.params)

    default_params = sat.resolve_lgb_params()
    default_params.update(resolve_device_params(args))

    default_test_runs = evaluate_final_on_test(
        run_bundle,
        default_params,
        final_seed_list,
        args.early_stopping_rounds,
        'default_lightgbm'
    )
    best_test_runs = evaluate_final_on_test(
        run_bundle,
        best_params,
        final_seed_list,
        args.early_stopping_rounds,
        'optuna_best_lightgbm'
    )
    test_runs = pd.concat([default_test_runs, best_test_runs], axis=0).reset_index(drop=True)

    summary_rows = []
    for setup_name, setup_df in test_runs.groupby('setup'):
        summary_rows.append({
            'setup': setup_name,
            'valid_qwk_mean': float(setup_df['valid_qwk'].mean()),
            'valid_macro_f1_mean': float(setup_df['valid_macro_f1'].mean()),
            'valid_accuracy_mean': float(setup_df['valid_accuracy'].mean()),
            'best_iteration_mean': float(setup_df['best_iteration'].mean()),
            'best_iteration_median': int(setup_df['best_iteration'].median()),
            'test_qwk_mean': float(setup_df['test_qwk'].mean()),
            'test_macro_f1_mean': float(setup_df['test_macro_f1'].mean()),
            'test_accuracy_mean': float(setup_df['test_accuracy'].mean()),
            'test_weighted_f1_mean': float(setup_df['test_weighted_f1'].mean()),
            'test_adjacent_error_rate_mean': float(setup_df['test_adjacent_error_rate'].mean()),
            'test_far_miss_rate_mean': float(setup_df['test_far_miss_rate'].mean())
        })
    summary_df = pd.DataFrame(summary_rows).sort_values('valid_qwk_mean', ascending=False).reset_index(drop=True)

    manifest = {
        'study_name': args.study_name,
        'input_path': args.input_path,
        'spec': args.spec,
        'numeric_scheme': args.numeric_scheme,
        'drop_2018': bool(args.drop_2018),
        'rows': {
            'train_2015_2020': int(len(run_bundle['train_df'])),
            'valid_2024': int(len(run_bundle['valid_df'])),
            'test_2025': int(len(run_bundle['test_df']))
        },
        'feature_columns': run_bundle['features'],
        'n_trials_requested': int(args.n_trials),
        'trials_completed': int(len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])),
        'objective_metric': args.objective_metric,
        'holdout_weight': float(args.holdout_weight),
        'rolling_weight': float(args.rolling_weight),
        'rolling_folds': int(len(run_bundle['rolling_bundles'])),
        'tuning_seed_list': tune_seed_list,
        'final_seed_list': final_seed_list,
        'rolling_seed_list': rolling_seed_list,
        'search_space': args.search_space,
        'device': 'gpu' if args.use_gpu else 'cpu',
        'gpu_backend': args.gpu_backend if args.use_gpu else None,
        'gpu_device_id': int(args.gpu_device_id) if args.use_gpu else None,
        'persist_study': bool(storage_cfg['persist_study']),
        'storage_path': str(storage_cfg['storage_path']) if storage_cfg['storage_path'] else None,
        'best_trial_number': int(study.best_trial.number),
        'best_trial_value': float(study.best_value),
        'best_trial_user_attrs': json_ready(study.best_trial.user_attrs),
        'best_params': json_ready(best_params),
        'default_params': json_ready(default_params)
    }

    trials_df.to_csv(output_paths['trials_path'], index=False)
    summary_df.to_csv(output_paths['summary_path'], index=False)
    test_runs.to_csv(output_paths['test_runs_path'], index=False)
    write_json(json_ready(best_params), output_paths['best_params_path'])
    write_json(json_ready(manifest), output_paths['manifest_path'])

    log_line('')
    log_line(f'[write] {output_paths["trials_path"]}')
    log_line(f'[write] {output_paths["summary_path"]}')
    log_line(f'[write] {output_paths["test_runs_path"]}')
    log_line(f'[write] {output_paths["best_params_path"]}')
    log_line(f'[write] {output_paths["manifest_path"]}')
    log_line('')
    log_line(
        f'[done] best_trial={study.best_trial.number} '
        f'objective={float(study.best_value):.4f} '
        f'best_valid_qwk={float(study.best_trial.user_attrs.get("holdout_qwk_mean", np.nan)):.4f}'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
