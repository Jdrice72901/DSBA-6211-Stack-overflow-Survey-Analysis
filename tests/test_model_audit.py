from pathlib import Path

import numpy as np
import pandas as pd

from src import comp_clean, model_audit

PARQUET_PATH = Path('data/derived/clean_core.parquet')


def test_clean_core_parquet_contract():
    assert PARQUET_PATH.exists()

    df = pd.read_parquet(PARQUET_PATH)

    required_cols = {
        'row_id',
        'survey_year',
        'country',
        'country_clean',
        'region',
        'employment_primary',
        'industry_clean',
        'ic_or_pm_clean',
        'years_code_pro_clean',
        'work_exp_clean',
        'comp',
        'comp_usd_clean',
        'comp_real_2025',
        'log_comp_real_2025',
        'is_comp_analysis_sample',
        'is_comp_model_core',
        'is_comp_model_tech_rich',
        'is_comp_model_ai_era',
        'is_comp_model_sample'
    }
    assert required_cols.issubset(df.columns)
    assert df['row_id'].is_unique
    assert set(df['survey_year'].dropna().unique()) == set(range(2015, 2026))

    assert df['comp_usd_clean'].dropna().between(1000, 1_000_000).all()
    assert np.allclose(
        df['log_comp_real_2025'].dropna().to_numpy(),
        np.log(df.loc[df['log_comp_real_2025'].notna(), 'comp_real_2025']).to_numpy()
    )

    assert (df['is_comp_model_core'] == (df['is_comp_analysis_sample'] & df['survey_year'].ge(2019))).all()
    assert (df['is_comp_model_tech_rich'] == (df['is_comp_analysis_sample'] & df['survey_year'].ge(2021))).all()
    assert (df['is_comp_model_ai_era'] == (df['is_comp_analysis_sample'] & df['survey_year'].ge(2023))).all()
    assert (df['is_comp_model_sample'] == df['is_comp_model_core']).all()
    assert df.loc[df['country_clean'].isna(), 'region'].isna().all()


def test_build_clean_core_single_year_smoke(monkeypatch):
    monkeypatch.setattr(comp_clean, 'YEARS', [2025], raising=False)

    clean = comp_clean.build_clean_core()

    assert set(clean['survey_year'].unique()) == {2025}
    assert clean['row_id'].is_unique
    assert clean.shape[0] > 0
    assert clean['comp_usd_clean'].dropna().between(1000, 1_000_000).all()
    assert (clean['is_comp_model_sample'] == clean['is_comp_analysis_sample']).all()
    assert clean['country_clean'].notna().any()
    assert clean['region'].notna().any()
    assert clean['employment_primary'].notna().any()
    assert clean['ic_or_pm_clean'].notna().any()


def test_validate_clean_core_passes_on_saved_parquet():
    df = pd.read_parquet(PARQUET_PATH)

    assert comp_clean.validate_clean_core(df) is True


def test_audit_helpers_return_expected_shapes():
    df = pd.read_parquet(PARQUET_PATH)
    audit = comp_clean.audit_clean_core(df)

    assert {
        'year_counts',
        'country_region',
        'numeric_masks',
        'comp_outliers',
        'unmapped_countries'
    }.issubset(set(audit))
    assert set(audit['year_counts']['survey_year']) == set(range(2015, 2026))
    assert {'field', 'clean_field', 'source_non_null', 'range_masked', 'clean_non_null'}.issubset(audit['numeric_masks'].columns)
    assert {'survey_year', 'comp_exact_cap', 'comp_model_rows'}.issubset(audit['comp_outliers'].columns)


def test_model_audit_hierarchical_median_and_job_sat_helpers():
    train = pd.DataFrame({
        'survey_year': [2019, 2019, 2020],
        'country_clean': ['United States', 'Germany', 'United States'],
        'region': ['Americas', 'Europe', 'Americas'],
        'comp_real_2025': [100_000.0, 80_000.0, 120_000.0],
        'log_comp_real_2025': np.log([100_000.0, 80_000.0, 120_000.0])
    })
    score = pd.DataFrame({
        'survey_year': [2024, 2024, 2024],
        'country_clean': ['United States', 'France', pd.NA],
        'region': ['Americas', 'Europe', 'Asia'],
        'comp_real_2025': [110_000.0, 90_000.0, 70_000.0],
        'log_comp_real_2025': np.log([110_000.0, 90_000.0, 70_000.0])
    })

    pred_log = model_audit.predict_hier_median(
        train,
        score,
        [
            ['country_clean'],
            ['region'],
            []
        ]
    )

    assert np.isclose(np.exp(pred_log.iloc[0]), 110_000.0, atol=10_000.0)
    assert np.isclose(np.exp(pred_log.iloc[1]), 80_000.0, atol=10_000.0)
    assert np.isclose(np.exp(pred_log.iloc[2]), 100_000.0, atol=20_000.0)

    sat = pd.DataFrame({
        'survey_year': [2017, 2018, 2025],
        'job_sat': ['8', 'Moderately dissatisfied', '10']
    })
    sat = model_audit.add_job_sat_std(sat)
    assert sat['job_sat_std'].tolist() == [4, 2, 5]


def test_build_job_sat_model_frame_filters_to_supported_years_and_employment():
    clean = pd.DataFrame({
        'survey_year': [2019, 2021, 2024, 2025],
        'job_sat': ['Very satisfied', pd.NA, '8', '10'],
        'is_professional': [True, True, False, True],
        'is_full_time_employed': [True, True, False, False],
        'is_part_time_employed': [False, False, False, False],
        'is_independent': [False, False, False, True],
        'employment_group': [
            'Employed full-time',
            'Employed full-time',
            'Student',
            'Independent / contract'
        ]
    })

    frame = model_audit.build_job_sat_model_frame(clean)

    assert frame['survey_year'].tolist() == [2019, 2025]
    assert frame['job_sat_std'].tolist() == [5, 5]


def test_build_job_sat_model_frame_uses_paid_work_flags_for_mixed_status_rows():
    clean = pd.DataFrame({
        'survey_year': [2025, 2025],
        'job_sat': ['10', '10'],
        'is_professional': [True, True],
        'is_full_time_employed': [False, False],
        'is_part_time_employed': [True, False],
        'is_independent': [False, False],
        'employment_group': ['Student', 'Student']
    })

    frame = model_audit.build_job_sat_model_frame(clean)

    assert frame.shape[0] == 1
