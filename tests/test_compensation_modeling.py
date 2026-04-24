from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import compensation_modeling

PARQUET_PATH = Path('data/derived/clean_core.parquet')


def make_comp_frame():
    rows = []
    countries = ['United States', 'Germany', 'India']
    regions = ['Americas', 'Europe', 'Asia']
    employment_groups = ['Employed full-time', 'Employed part-time', 'Independent / contract']
    educations = ["Bachelor's degree", "Master's degree", 'Professional degree']
    org_sizes = ['1-19', '20-99', '100-999']
    industries = ['Software / IT', 'Financial services', 'Manufacturing / logistics']
    ic_or_pm = ['Individual contributor', 'People manager', 'Individual contributor']
    remotes = ['Mostly in-person', 'Hybrid', 'Mostly remote']
    ai_uses = ['Yes', 'No', 'Yes']
    ai_sents = ['Positive', 'Neutral', 'Negative']
    languages = ['Python;JavaScript', 'Python;Rust', 'SQL;Go']
    databases = ['PostgreSQL;Redis', 'Redis;MongoDB', 'PostgreSQL;MongoDB']
    platforms = ['AWS;Docker', 'AWS;Azure', 'Azure;GCP']
    webframes = ['React;Django', 'Vue.js', 'Angular']
    misc_tech = ['Docker;Kubernetes', 'Docker', 'Kubernetes']
    learn_code = ['Books;School', 'Books;Courses', 'Documentation;School']
    learn_code_online = ['Stack Overflow;YouTube', 'YouTube', 'Stack Overflow']
    coding_activities = ['Professional development;Hobby', 'Professional development', 'Hobby']
    op_sys_prof = ['Windows;Linux', 'Linux', 'macOS']
    role_families = ['Full-stack', 'Back-end', 'Data / ML']

    for year in range(2019, 2026):
        for idx in range(3):
            comp_real = 80_000 + (year - 2019) * 7_500 + idx * 2_500
            rows.append({
                'survey_year': year,
                'country_clean': countries[idx],
                'region': regions[idx],
                'is_professional': True,
                'is_paid_worker': True,
                'employment_primary': employment_groups[idx],
                'is_full_time_employed': idx == 0,
                'is_part_time_employed': idx == 1,
                'is_independent': idx == 2,
                'education_clean': educations[idx],
                'org_size_clean': org_sizes[idx],
                'industry_clean': industries[idx],
                'ic_or_pm_clean': ic_or_pm[idx],
                'remote_group': remotes[idx],
                'ai_use': ai_uses[idx] if year >= 2023 else pd.NA,
                'ai_sent': ai_sents[idx] if year >= 2023 else pd.NA,
                'age_mid': str(25 + idx),
                'years_code_pro_clean': str(4 + idx),
                'work_exp_clean': str(5 + idx),
                'professional_experience_years': str(4 + idx),
                'role_family': role_families[idx],
                'language': languages[idx],
                'database': databases[idx],
                'platform': platforms[idx],
                'webframe': webframes[idx],
                'misc_tech': misc_tech[idx],
                'learn_code': learn_code[idx],
                'learn_code_online': learn_code_online[idx],
                'coding_activities': coding_activities[idx],
                'op_sys_prof': op_sys_prof[idx],
                'comp_usd_clean': str(comp_real),
                'comp_real_2025': str(comp_real)
            })

    return pd.DataFrame(rows)


def test_coerce_compensation_frame_normalizes_numeric_fields_and_year_string():
    frame = make_comp_frame().iloc[:2].copy()
    frame['role_family'] = ['Full-stack', 'Back-end']
    frame['language'] = ['Python;JavaScript;Rust;Go', 'not a number']

    coerced = compensation_modeling.coerce_compensation_frame(frame)

    assert coerced['survey_year_str'].tolist() == ['2019', '2019']
    assert pd.api.types.is_numeric_dtype(coerced['language_count'])
    assert pd.api.types.is_numeric_dtype(coerced[compensation_modeling.TARGET_COL])
    assert np.isclose(coerced['role_full_stack'].iloc[0], 1.0)
    assert coerced['language_count'].iloc[0] == 4
    assert coerced['language_count'].iloc[1] == 1


def test_add_top_tech_flags_uses_fit_frame_and_fills_expected_columns():
    fit_frame = pd.DataFrame({
        'language': ['Python;JavaScript', 'Python;Rust', 'SQL;Go'],
        'database': ['PostgreSQL;Redis', 'Redis;MongoDB', 'MongoDB;PostgreSQL'],
        'platform': ['AWS;Docker', 'AWS;Azure', 'Azure;GCP']
    })
    frame = pd.DataFrame({
        'language': ['Python;Rust', 'Go;SQL', None],
        'database': ['Redis', 'MongoDB', None],
        'platform': ['AWS', 'GCP', None]
    })

    flagged, created = compensation_modeling.add_top_tech_flags(
        frame,
        top_n_map={'language': 2, 'database': 1, 'platform': 1},
        fit_frame=fit_frame
    )

    assert set(created) == {
        'language_python',
        'language_javascript',
        'database_postgresql',
        'platform_aws'
    }
    assert flagged['language_python'].tolist() == [1, 0, 0]
    assert flagged['database_postgresql'].tolist() == [0, 0, 0]
    assert flagged['platform_aws'].tolist() == [1, 0, 0]


def test_tech_flag_name_avoids_common_token_collisions():
    assert compensation_modeling.tech_flag_name('language', 'C') == 'language_c'
    assert compensation_modeling.tech_flag_name('language', 'C++') == 'language_c_plusplus'
    assert compensation_modeling.tech_flag_name('language', 'C#') == 'language_c_sharp'


def test_add_top_tech_flags_matches_full_tokens_not_substrings():
    fit_frame = pd.DataFrame({
        'language': ['C;Rust', 'C++;SQL', 'C++;Go', 'C#;Rust', 'C#;Go']
    })
    frame = pd.DataFrame({
        'language': ['C++', 'C#', 'Objective-C', 'C']
    })

    flagged, created = compensation_modeling.add_top_tech_flags(
        frame,
        top_n_map={'language': 7},
        fit_frame=fit_frame
    )

    assert {'language_c', 'language_c_plusplus', 'language_c_sharp'}.issubset(set(created))
    assert flagged['language_c'].tolist() == [0, 0, 0, 1]
    assert flagged['language_c_plusplus'].tolist() == [1, 0, 0, 0]
    assert flagged['language_c_sharp'].tolist() == [0, 1, 0, 0]


def test_prepare_lgbm_frame_and_winsor_targets_support_small_frames():
    frame = make_comp_frame().iloc[:9].copy()
    frame = compensation_modeling.add_winsor_targets(frame)

    assert compensation_modeling.REAL_WINSOR_COL in frame.columns
    assert compensation_modeling.WINSOR_TARGET_COL in frame.columns
    assert np.isfinite(frame[compensation_modeling.WINSOR_TARGET_COL]).all()

    prepared, num_imputer = compensation_modeling.prepare_lgbm_frame(
        frame,
        ['country_clean', 'region'],
        ['age_mid', 'professional_experience_years', 'language_count']
    )

    assert str(prepared['country_clean'].dtype) == 'category'
    assert str(prepared['region'].dtype) == 'category'
    assert pd.api.types.is_numeric_dtype(prepared['age_mid'])
    assert num_imputer.statistics_.shape[0] == 3


def test_build_compensation_bundle_runs_on_saved_parquet():
    clean_core = pd.read_parquet(PARQUET_PATH)

    bundle = compensation_modeling.build_compensation_bundle(clean_core)

    assert {'core_df', 'tech_df', 'ai_df', 'role_cols', 'feature_sets'} == set(bundle)
    assert bundle['core_df']['survey_year'].min() >= 2019
    assert bundle['tech_df']['survey_year'].min() >= 2021
    assert bundle['ai_df']['survey_year'].min() >= 2023
    assert len(bundle['role_cols']) > 0


def test_compare_same_sample_setups_runs_on_synthetic_data():
    frame = make_comp_frame()
    result = compensation_modeling.compare_same_sample_setups(
        frame,
        lgb_params={
            'n_estimators': 20,
            'learning_rate': 0.1,
            'num_leaves': 7,
            'min_child_samples': 1
        }
    )

    summary = result['summary']

    assert {
        'Country-region median baseline',
        'Ridge core',
        'LightGBM core',
        'LightGBM core + top tech flags',
        'LightGBM winsorized target + top tech flags',
        'LightGBM tech-rich window',
        'LightGBM AI-era window'
    }.issubset(set(summary['setup']))
    assert np.isfinite(summary['valid_medae_real']).all()
    assert np.isfinite(summary['test_medae_real']).all()
    assert set(result['bundle']) == {
        'core_df',
        'tech_df',
        'ai_df',
        'role_cols',
        'feature_sets'
    }
    assert result['selected_main_setup'] == 'LightGBM winsorized target + top tech flags'
    assert np.isfinite(result['selected_main_result']['test_metrics']['medae_real'])


def test_fit_selected_compensation_model_runs_on_synthetic_data():
    frame = make_comp_frame()
    result = compensation_modeling.fit_selected_compensation_model(
        frame,
        lgb_params={
            'n_estimators': 20,
            'learning_rate': 0.1,
            'num_leaves': 7,
            'min_child_samples': 1
        }
    )

    assert result['setup'] == 'LightGBM winsorized target + top tech flags'
    assert result['train_years'] == compensation_modeling.CORE_WINDOW_YEARS
    assert result['valid_year'] == compensation_modeling.VALID_YEAR
    assert result['test_year'] == compensation_modeling.TEST_YEAR
    assert len(result['tech_flag_cols']) > 0
    assert np.isfinite(result['valid_metrics']['medae_real'])
    assert np.isfinite(result['test_metrics']['medae_real'])


def test_score_prediction_frame_and_feature_family_helpers():
    frame = pd.DataFrame({
        compensation_modeling.TARGET_COL: np.log([50_000, 80_000]),
        'region': ['Americas', 'Europe']
    })
    scored = compensation_modeling.score_prediction_frame(
        frame,
        np.log([55_000, 70_000])
    )

    assert {'pred_real', 'actual_real', 'signed_error_real', 'abs_error_real', 'pct_error'}.issubset(scored.columns)
    assert compensation_modeling.feature_family('country_clean') == 'Geography'
    assert compensation_modeling.feature_family('language_python') == 'Top tech flags'
    assert compensation_modeling.feature_family('professional_experience_years') == 'Experience and age'


def test_build_report_paths_and_report_bundle_run_on_synthetic_data():
    frame = make_comp_frame()
    paths = compensation_modeling.build_report_paths(compensation_modeling.OUTPUT_ROOT / 'comp_reporting_test')

    assert paths['output_dir'].exists()
    assert 'context' in paths['figures']
    assert 'setup_summary' in paths['tables']

    report = compensation_modeling.build_compensation_report_bundle(
        frame,
        lgb_params={
            'n_estimators': 20,
            'learning_rate': 0.1,
            'num_leaves': 7,
            'min_child_samples': 1
        },
        include_shap=False,
        include_rolling=False
    )

    assert np.isfinite(report['selected_scored']['abs_error_real']).all()
    assert not report['region_metrics'].empty
    assert not report['country_metrics'].empty
    assert not report['decile_compare'].empty
    assert report['decile_compare'].groupby('setup')['rows'].sum().gt(0).all()
    assert report['shap_bundle'] is None
    assert report['rolling'] is None


def test_build_us_report_paths_and_bundle_run_on_synthetic_data():
    frame = make_comp_frame()
    paths = compensation_modeling.build_us_report_paths(compensation_modeling.OUTPUT_ROOT / 'comp_us_reporting_test')

    assert paths['output_dir'].exists()
    assert 'compare' in paths['figures']
    assert 'summary' in paths['tables']

    report = compensation_modeling.build_us_compensation_report_bundle(
        frame,
        country='United States',
        lgb_params={
            'n_estimators': 20,
            'learning_rate': 0.1,
            'num_leaves': 7,
            'min_child_samples': 1
        },
        shap_sample_size=10,
        include_shap=True
    )

    assert set(report['summary']['model_view']) == {
        compensation_modeling.GLOBAL_MAIN_VIEW,
        compensation_modeling.US_REFIT_VIEW
    }
    assert len(report['global_test_scored']) >= len(report['country_test_scored'])
    assert len(report['predictions']) == len(report['global_test_scored']) + len(report['country_test_scored'])
    assert set(report['predictions']['model_view']) == {
        compensation_modeling.GLOBAL_MAIN_VIEW,
        compensation_modeling.US_REFIT_VIEW
    }
    assert set(report['shap_compare']['model_view']) == {
        compensation_modeling.GLOBAL_MAIN_VIEW,
        compensation_modeling.US_REFIT_VIEW
    }
    assert 'country_clean' not in report['country_result']['feature_cols']
    assert 'region' not in report['country_result']['feature_cols']
    assert report['shap_compare'] is not None
    assert not report['shap_top_features'].empty


def test_report_plot_helpers_write_nonempty_pngs():
    frame = make_comp_frame()
    plt.switch_backend('Agg')
    report = compensation_modeling.build_compensation_report_bundle(
        frame,
        lgb_params={
            'n_estimators': 20,
            'learning_rate': 0.1,
            'num_leaves': 7,
            'min_child_samples': 1
        },
        include_shap=False,
        include_rolling=False
    )
    report['rolling'] = {
        'results': [
            {
                'setup': 'Ridge core alpha=25.0',
                'folds': pd.DataFrame({
                    'valid_year': [2024, 2025],
                    'medae_real': [15_000, 16_000],
                    'r2_log': [0.42, 0.45]
                })
            },
            {
                'setup': 'LightGBM core + top tech flags [tiny]',
                'folds': pd.DataFrame({
                    'valid_year': [2024, 2025],
                    'medae_real': [14_000, 15_500],
                    'r2_log': [0.47, 0.5]
                })
            }
        ]
    }

    plot_dir = compensation_modeling.OUTPUT_ROOT / 'comp_plot_test_artifacts'
    plot_dir.mkdir(parents=True, exist_ok=True)
    context_path = plot_dir / 'context.png'
    diagnostics_path = plot_dir / 'diagnostics.png'
    rolling_path = plot_dir / 'rolling.png'

    compensation_modeling.plot_compensation_context(report, context_path)
    compensation_modeling.plot_test_diagnostics(report, diagnostics_path)
    compensation_modeling.plot_rolling_origin(report, rolling_path)

    assert context_path.exists() and context_path.stat().st_size > 0
    assert diagnostics_path.exists() and diagnostics_path.stat().st_size > 0
    assert rolling_path.exists() and rolling_path.stat().st_size > 0


def test_us_report_plot_helpers_and_writer_write_nonempty_artifacts():
    frame = make_comp_frame()
    plt.switch_backend('Agg')
    report = compensation_modeling.build_us_compensation_report_bundle(
        frame,
        country='United States',
        lgb_params={
            'n_estimators': 20,
            'learning_rate': 0.1,
            'num_leaves': 7,
            'min_child_samples': 1
        },
        shap_sample_size=10,
        include_shap=True
    )
    plot_dir = compensation_modeling.OUTPUT_ROOT / 'comp_us_plot_test_artifacts'
    plot_dir.mkdir(parents=True, exist_ok=True)
    compare_path = plot_dir / 'compare.png'
    diagnostics_path = plot_dir / 'diagnostics.png'
    feature_shift_path = plot_dir / 'feature_shift.png'
    shap_beeswarm_path = plot_dir / 'shap_beeswarm.png'

    compensation_modeling.plot_us_compare(report, compare_path)
    compensation_modeling.plot_us_diagnostics(report, diagnostics_path)
    compensation_modeling.plot_us_feature_shift(report, feature_shift_path)
    compensation_modeling.plot_us_shap_beeswarm(report, shap_beeswarm_path)

    assert compare_path.exists() and compare_path.stat().st_size > 0
    assert diagnostics_path.exists() and diagnostics_path.stat().st_size > 0
    assert feature_shift_path.exists() and feature_shift_path.stat().st_size > 0
    assert shap_beeswarm_path.exists() and shap_beeswarm_path.stat().st_size > 0

    output_dir = compensation_modeling.OUTPUT_ROOT / 'comp_us_report_test_artifacts'
    written = compensation_modeling.generate_us_compensation_report(
        frame,
        output_dir=output_dir,
        country='United States',
        shap_sample_size=10
    )

    assert written['paths']['manifest'].exists()
    assert written['paths']['tables']['summary'].exists()
    assert written['paths']['tables']['test_predictions'].exists()
    assert written['paths']['tables']['shap_compare'].exists()
    assert written['paths']['tables']['shap_top_features'].exists()
    assert written['paths']['figures']['compare'].exists()
    assert written['paths']['figures']['diagnostics'].exists()
    assert written['paths']['figures']['feature_shift'].exists()
    assert written['paths']['figures']['shap_beeswarm'].exists()


def test_rolling_origin_setup_comparison_runs_on_synthetic_data():
    frame = make_comp_frame()
    result = compensation_modeling.rolling_origin_setup_comparison(
        frame,
        ridge_alphas=[25.0],
        lgb_presets={
            'tiny': {
                'n_estimators': 20,
                'learning_rate': 0.1,
                'num_leaves': 7,
                'min_child_samples': 1
            }
        }
    )

    summary = result['summary']
    assert 'Ridge core alpha=25.0' in set(summary['setup'])
    assert 'LightGBM core + top tech flags [tiny]' in set(summary['setup'])
    assert np.isfinite(summary['mean_valid_medae_real']).all()
