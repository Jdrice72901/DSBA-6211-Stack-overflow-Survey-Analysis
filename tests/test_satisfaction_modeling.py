from pathlib import Path

import numpy as np
import pandas as pd

from src import model_audit, satisfaction_modeling

PARQUET_PATH = Path('data/derived/clean_core.parquet')


def sat_raw_value(year, label):
    five_point = {
        1: "I hate my job",
        2: "I'm somewhat dissatisfied with my job",
        3: "I'm neither satisfied nor dissatisfied with my job",
        4: "I'm somewhat satisfied with my job",
        5: "I love my job"
    }
    seven_point = {
        1: 'Extremely dissatisfied',
        2: 'Slightly dissatisfied',
        3: 'Neither satisfied nor dissatisfied',
        4: 'Slightly satisfied',
        5: 'Extremely satisfied'
    }
    numeric = {
        1: '0',
        2: '3',
        3: '5',
        4: '8',
        5: '10'
    }

    if year in {2017, 2024, 2025}:
        return numeric[label]
    if year == 2018:
        return seven_point[label]
    return five_point[label]


def make_sat_frame(rows_per_label=4):
    rows = []
    years = satisfaction_modeling.SAT_CANONICAL_YEARS
    countries = ['United States', 'Germany', 'India', 'Canada', 'Brazil']
    regions = ['Americas', 'Europe', 'Asia', 'Americas', 'Americas']
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64']
    educations = [
        "Bachelor's degree",
        "Master's degree",
        'Professional degree',
        'Secondary school',
        'Some college'
    ]
    org_sizes = ['1-19', '20-99', '100-999', '1,000-4,999', '10,000 or more']
    remote_groups = ['Mostly in-person', 'Hybrid', 'Mostly remote', 'Hybrid', 'Mostly remote']

    for year in years:
        for label in satisfaction_modeling.SAT_LABELS:
            for rep in range(rows_per_label):
                idx = (label + rep) % len(countries)
                rows.append({
                    'survey_year': year,
                    'job_sat': sat_raw_value(year, label),
                    'is_professional': True,
                    'is_full_time_employed': True,
                    'is_part_time_employed': False,
                    'is_independent': False,
                    'employment_primary': 'Employed full-time',
                    'employment_group': 'Employed full-time',
                    'country_clean': countries[idx],
                    'region': regions[idx],
                    'age_group': age_groups[label - 1],
                    'education_clean': educations[idx],
                    'org_size_clean': org_sizes[idx],
                    'remote_group': remote_groups[idx],
                    'years_code_clean': 2 * label + rep,
                    'language_count': label + rep % 2,
                    'database_count': label,
                    'platform_count': max(1, label - 1),
                    'role_family_count': 1 + (label % 3),
                    'role_front_end': int(label <= 2),
                    'role_back_end': int(label in {3, 4}),
                    'role_data_ml': int(label == 5),
                    'log_comp_real_2025': np.log(45_000 + label * 7_500 + rep * 500)
                })

    return pd.DataFrame(rows)


def test_standardize_job_sat_value_handles_text_numeric_and_alt_scheme():
    assert satisfaction_modeling.standardize_job_sat_value('10', 2025) == 5
    assert satisfaction_modeling.standardize_job_sat_value('3', 2024) == 2
    assert satisfaction_modeling.standardize_job_sat_value('3', 2024, numeric_scheme='alt_equal_width') == 2
    assert satisfaction_modeling.standardize_job_sat_value('8', 2017) == 4
    assert satisfaction_modeling.standardize_job_sat_value('Extremely dissatisfied', 2018) == 1
    assert satisfaction_modeling.standardize_job_sat_value("I love my job", 2019) == 5


def test_add_career_stage_features_handles_missing_inputs():
    frame = pd.DataFrame({
        'survey_year': [2019, 2020],
        'job_sat': ["I love my job", "I'm somewhat satisfied with my job"]
    })

    out = satisfaction_modeling.add_career_stage_features(frame)

    for col in ['age_mid', 'professional_experience_years', 'career_start_age_est', 'coding_start_age_est', 'pro_to_total_code_ratio']:
        assert col in out.columns
        assert out[col].isna().all()


def test_build_satisfaction_frame_filters_supported_years_and_adds_metadata():
    clean = pd.DataFrame({
        'survey_year': [2018, 2019, 2021, 2024, 2025],
        'job_sat': ['Extremely satisfied', 'I love my job', pd.NA, '8', '10'],
        'is_professional': [True, True, True, False, True],
        'is_full_time_employed': [False, True, True, False, False],
        'is_part_time_employed': [False, False, False, False, False],
        'is_independent': [False, False, False, False, True],
        'employment_primary': ['Student', 'Employed full-time', 'Employed full-time', 'Student', 'Independent / contract'],
        'employment_group': ['Student', 'Employed full-time', 'Employed full-time', 'Student', 'Independent / contract']
    })

    frame = satisfaction_modeling.build_satisfaction_frame(clean)

    assert frame['survey_year'].tolist() == [2019, 2025]
    assert frame[satisfaction_modeling.SAT_TARGET_COL].tolist() == [5, 5]
    assert frame[satisfaction_modeling.SAT_BINARY_COL].tolist() == [1, 1]
    assert frame[satisfaction_modeling.SAT_INSTRUMENT_COL].tolist() == ['text_5pt', 'numeric_11pt']
    assert frame['survey_year_str'].tolist() == ['2019', '2025']


def test_prepare_lgbm_frame_keeps_all_missing_numeric_columns():
    frame = pd.DataFrame({
        'survey_year': [2019, 2020],
        'survey_year_str': ['2019', '2020'],
        'region': ['Americas', 'Europe'],
        'country_clean': ['United States', 'Germany'],
        'age_group': ['25-34', '35-44'],
        'education_clean': ["Bachelor's degree", "Master's degree"],
        'employment_primary': ['Employed full-time', 'Employed full-time'],
        'org_size_clean': ['20-99', '100-999'],
        'remote_group': ['Hybrid', 'Mostly remote'],
        'years_code_clean': [5, 10],
        'language_count': [2, 3],
        'database_count': [1, 2],
        'platform_count': [1, 1],
        'role_family_count': [1, 2],
        'career_start_age_est': [pd.NA, pd.NA],
        'job_sat_std': [4, 5]
    })

    prepped, num_prep = satisfaction_modeling.prepare_lgbm_frame(
        frame,
        cat_cols=['survey_year_str', 'region', 'country_clean', 'age_group', 'education_clean', 'employment_primary', 'org_size_clean', 'remote_group'],
        num_cols=['years_code_clean', 'career_start_age_est']
    )
    scored = satisfaction_modeling.transform_lgbm_frame(
        frame,
        cat_cols=['survey_year_str', 'region', 'country_clean', 'age_group', 'education_clean', 'employment_primary', 'org_size_clean', 'remote_group'],
        num_cols=['years_code_clean', 'career_start_age_est'],
        num_prep=num_prep
    )

    assert list(prepped[['years_code_clean', 'career_start_age_est']].columns) == ['years_code_clean', 'career_start_age_est']
    assert prepped['career_start_age_est'].eq(0.0).all()
    assert scored['career_start_age_est'].eq(0.0).all()


def test_feature_availability_by_year_handles_missing_columns():
    frame = make_sat_frame(rows_per_label=1)
    availability = satisfaction_modeling.feature_availability_by_year(
        frame,
        ['country_clean', 'remote_group', 'missing_col']
    )

    assert set(availability.columns) == {'survey_year', 'rows', 'country_clean', 'remote_group', 'missing_col'}
    assert availability['country_clean'].eq(1.0).all()
    assert availability['missing_col'].isna().all()


def test_compare_satisfaction_setups_runs_on_synthetic_data():
    frame = make_sat_frame(rows_per_label=3)
    result = satisfaction_modeling.compare_satisfaction_setups(
        frame,
        lgb_params={
            'n_estimators': 25,
            'learning_rate': 0.1,
            'num_leaves': 15,
            'min_child_samples': 5
        },
        catboost_params={
            'iterations': 30,
            'learning_rate': 0.1,
            'depth': 4,
            'verbose': False
        }
    )

    summary = result['summary']

    assert {
        'Global mode baseline',
        'Country-region mode baseline',
        'Ordered baseline',
        'LightGBM core_no_comp',
        'CatBoost core_no_comp'
    }.issubset(set(summary['setup']))
    assert np.isfinite(summary['valid_qwk']).all()
    assert np.isfinite(summary['test_qwk']).all()
    assert result['selected_main_setup'] in {'LightGBM core_no_comp', 'CatBoost core_no_comp'}
    assert result['selected_family'] in {'lightgbm', 'catboost'}
    assert result['selected_family_comp_subset_setup'] in set(summary['setup'])
    assert result['selected_family_with_comp_setup'] in set(summary['setup'])
    assert not result['frame_counts'].empty
    assert not result['with_comp_counts'].empty
    assert result['selected_family_comp_subset_no_comp'] is not None
    assert result['selected_family_with_comp'] is not None


def test_subgroup_metrics_handles_missing_group_labels():
    frame = pd.DataFrame({
        'region': ['Americas', pd.NA, 'Europe'],
        'country_clean': ['United States', 'Unknown', 'Germany']
    })
    y_true = pd.Series([5, 4, 3], index=frame.index)
    y_pred = pd.Series([5, 3, 3], index=frame.index)

    result = satisfaction_modeling.subgroup_metrics(frame, y_true, y_pred, ['region'])

    assert result['rows'].sum() == 3
    assert set(result['region'].astype('string').fillna('Missing')) == {'Americas', 'Europe', 'Missing'}


def test_build_satisfaction_frame_real_data_contract():
    clean_core = pd.read_parquet(PARQUET_PATH)
    frame = satisfaction_modeling.build_satisfaction_frame(clean_core)

    assert set(frame['survey_year'].unique()) == set(satisfaction_modeling.SAT_CANONICAL_YEARS)
    assert frame[satisfaction_modeling.SAT_TARGET_COL].between(1, 5).all()
    assert model_audit.paid_work_mask(frame).all()
    assert frame['is_professional'].all()
    assert frame.loc[frame['survey_year'].eq(2025)].shape[0] > 0
