import math

import numpy as np
import pandas as pd

from src import comp_clean


def test_parse_midpoint_handles_numeric_ranges_and_text():
    assert comp_clean.parse_midpoint(7) == 7.0
    assert comp_clean.parse_midpoint(7.5) == 7.5
    assert comp_clean.parse_midpoint('10-20') == 15.0
    assert comp_clean.parse_midpoint('1,000 to 2,000') == 1500.0
    assert comp_clean.parse_midpoint('under 1 year') == 0.5
    assert comp_clean.parse_midpoint('Less than 1 year') == 0.5
    assert comp_clean.parse_midpoint('Younger than 5 years') == 2.5
    assert comp_clean.parse_midpoint('<10') == 5.0
    assert math.isnan(comp_clean.parse_midpoint('not sure'))
    assert math.isnan(comp_clean.parse_midpoint(pd.NA))


def test_clean_multiselect_keeps_order_dedupes_and_maps():
    value = 'Python; JavaScript; Python; C++; ReactJS; '
    cleaned = comp_clean.clean_multiselect(
        value,
        keep={'Python', 'JavaScript', 'C++'},
        remove=comp_clean.EXTRAS,
        replace=comp_clean.LANG_MAP
    )

    assert cleaned == 'Python;JavaScript;C++'


def test_clean_multiselect_returns_missing_when_everything_is_filtered_out():
    cleaned = comp_clean.clean_multiselect(
        'ReactJS;Slack;Other(s):',
        remove=comp_clean.EXTRAS
    )

    assert pd.isna(cleaned)


def test_clean_education_groups_known_variants():
    assert comp_clean.clean_education("Bachelor's degree in computer science") == "Bachelor's degree"
    assert comp_clean.clean_education('Some college/university study without earning a degree') == 'Some college/university'
    assert comp_clean.clean_education('Professional degree (JD, MD, etc.)') == 'Professional degree'
    assert comp_clean.clean_education('Self-taught') == 'No formal education / other'
    assert pd.isna(comp_clean.clean_education('Prefer not to say'))


def test_clean_org_size_groups_common_ranges():
    assert comp_clean.clean_org_size('Just me - I am a freelancer, sole proprietor, etc.') == 'Self-employed'
    assert comp_clean.clean_org_size('1-4 employees') == '1-19'
    assert comp_clean.clean_org_size('20 to 99 employees') == '20-99'
    assert comp_clean.clean_org_size('100 to 499 employees') == '100-999'
    assert comp_clean.clean_org_size('1,000 to 4,999 employees') == '1,000-9,999'
    assert comp_clean.clean_org_size('10,000 or more employees') == '10,000+'
    assert pd.isna(comp_clean.clean_org_size('Don\'t know'))
    assert pd.isna(comp_clean.clean_org_size('I don’t know'))


def test_clean_ic_or_pm_standardizes_later_wave_labels():
    assert comp_clean.clean_ic_or_pm('Independent contributor') == 'Individual contributor'
    assert comp_clean.clean_ic_or_pm('Individual contributor') == 'Individual contributor'
    assert comp_clean.clean_ic_or_pm('People manager') == 'People manager'
    assert pd.isna(comp_clean.clean_ic_or_pm('Prefer not to say'))


def test_clean_industry_maps_later_and_legacy_variants():
    assert comp_clean.clean_industry('Software Development', 2025) == 'Software / IT'
    assert comp_clean.clean_industry('Information Services, IT, Software Development, or other Technology', 2023) == 'Software / IT'
    assert comp_clean.clean_industry('Fintech', 2024) == 'Financial services'
    assert comp_clean.clean_industry('Manufacturing, Transportation, or Supply Chain', 2023) == 'Manufacturing / logistics'
    assert comp_clean.clean_industry('Consulting', 2016) == 'Professional services'
    assert pd.isna(comp_clean.clean_industry('Government agency or public school/university', 2017))


def test_group_remote_collapses_reported_variants():
    assert comp_clean.group_remote('Full in-person') == 'Mostly in-person'
    assert comp_clean.group_remote('About half the time') == 'Hybrid'
    assert comp_clean.group_remote('Fully remote') == 'Mostly remote'
    assert pd.isna(comp_clean.group_remote('Occasionally remote'))


def test_role_family_maps_common_job_titles():
    assert comp_clean.role_family('Full-stack developer') == 'Full-stack'
    assert comp_clean.role_family('Back-end engineer') == 'Back-end'
    assert comp_clean.role_family('Front-end web developer') == 'Front-end'
    assert comp_clean.role_family('Machine learning engineer') == 'Data / ML'
    assert comp_clean.role_family('Android developer') == 'Mobile'
    assert comp_clean.role_family('Site reliability engineer') == 'DevOps / Cloud'
    assert comp_clean.role_family('Developer, embedded applications or devices') == 'Embedded / hardware'
    assert comp_clean.role_family('Security professional') == 'Security'
    assert comp_clean.role_family('Architect, software or solutions') == 'Architecture'
    assert comp_clean.role_family('QA tester') == 'QA / Testing'
    assert comp_clean.role_family('Student') == 'Student / Academic'
    assert comp_clean.role_family('Something else entirely') == 'Other'


def test_build_role_family_value_dedupes_and_preserves_role_order():
    value = 'Full-stack developer;Back-end engineer;Full-stack developer;QA tester'

    assert comp_clean.build_role_family_value(value) == 'Full-stack;Back-end;QA / Testing'


def test_add_role_family_features_expands_flags_from_compact_role_field():
    frame = pd.DataFrame({'role_family': ['Full-stack;Back-end', pd.NA]})

    out = comp_clean.add_role_family_features(frame)

    assert out['role_family_count'].iloc[0] == 2.0
    assert np.isnan(out['role_family_count'].iloc[1])
    assert out['role_full_stack'].tolist() == [1, 0]
    assert out['role_back_end'].tolist() == [1, 0]
