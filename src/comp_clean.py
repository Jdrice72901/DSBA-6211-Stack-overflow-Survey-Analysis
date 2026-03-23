"""
This file takes the findings from the Cleaning.ipynb and consolidates them into a
single, repeatable process to build a cleaned set of data ready for compensation
modeling. Creates a parquet output under data/derived/clean_core.parquet
"""

import logging
import re
import warnings
from pathlib import Path

import country_converter as coco
import numpy as np
import pandas as pd

# =====================================================================================
# Setup
# =====================================================================================

warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('country_converter').setLevel(logging.ERROR)
logging.getLogger('country_converter.country_converter').setLevel(logging.ERROR)

# Robust method to find repo root and define directory paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
DERIVED_DIR = DATA_DIR / 'derived'
OUT_PATH = DERIVED_DIR / 'clean_core.parquet'

# Fields we intend to collapse the varied yearly survey questions down to
RAW_FIELDS = [
    'response_id',
    'country',
    'age',
    'gender',
    'ethnicity',
    'main_branch',
    'student',
    'employment',
    'education',
    'undergrad_major',
    'age_first_code',
    'org_size',
    'dev_type',
    'industry',
    'remote',
    'years_code',
    'years_code_pro',
    'work_exp',
    'job_seek',
    'work_week_hrs',
    'learn_code',
    'learn_code_online',
    'coding_activities',
    'comp',
    'job_sat',
    'language',
    'database',
    'platform',
    'webframe',
    'misc_tech',
    'op_sys_prof',
    'ai_use',
    'ai_sent'
]

# Mapping of columns names of desired fields for each year, where they exist
YEAR_INFO = {
    2015: {
        'file': '2015 Stack Overflow Developer Survey Responses.csv',
        'header': 1,
        'response_id': None,
        'country': 'Country',
        'age': 'Age',
        'gender': 'Gender',
        'ethnicity': None,
        'main_branch': None,
        'student': None,
        'employment': 'Employment Status',
        'education': None,
        'undergrad_major': None,
        'age_first_code': None,
        'org_size': None,
        'dev_type': 'Occupation',
        'industry': 'Industry',
        'remote': 'Remote Status',
        'years_code': 'Years IT / Programming Experience',
        'years_code_pro': None,
        'work_exp': None,
        'job_seek': None,
        'work_week_hrs': None,
        'learn_code': None,
        'learn_code_online': None,
        'coding_activities': None,
        'comp': 'Compensation: midpoint',
        'job_sat': 'Job Satisfaction',
        'language': None,
        'database': None,
        'platform': None,
        'webframe': None,
        'misc_tech': None,
        'op_sys_prof': None,
        'ai_use': None,
        'ai_sent': None
    },
    2016: {
        'file': '2016 Stack Overflow Survey Responses.csv',
        'header': 0,
        'response_id': None,
        'country': 'country',
        'age': 'age_midpoint',
        'gender': 'gender',
        'ethnicity': None,
        'main_branch': None,
        'student': None,
        'employment': 'employment_status',
        'education': 'education',
        'undergrad_major': None,
        'age_first_code': None,
        'org_size': 'company_size_range',
        'dev_type': 'occupation',
        'industry': 'industry',
        'remote': 'remote',
        'years_code': 'experience_range',
        'years_code_pro': 'experience_range',
        'work_exp': None,
        'job_seek': None,
        'work_week_hrs': None,
        'learn_code': None,
        'learn_code_online': None,
        'coding_activities': None,
        'comp': 'salary_midpoint',
        'job_sat': 'job_satisfaction',
        'language': 'tech_do',
        'database': None,
        'platform': None,
        'webframe': None,
        'misc_tech': None,
        'op_sys_prof': None,
        'ai_use': None,
        'ai_sent': None
    },
    2017: {
        'file': 'survey_results_public2017.csv',
        'header': 0,
        'response_id': 'Respondent',
        'country': 'Country',
        'age': None,
        'gender': 'Gender',
        'ethnicity': None,
        'main_branch': 'Professional',
        'student': 'Professional',
        'employment': 'EmploymentStatus',
        'education': 'FormalEducation',
        'undergrad_major': 'MajorUndergrad',
        'age_first_code': None,
        'org_size': 'CompanySize',
        'dev_type': 'DeveloperType',
        'industry': 'CompanyType',
        'remote': 'HomeRemote',
        'years_code': 'YearsProgram',
        'years_code_pro': 'YearsCodedJob',
        'work_exp': None,
        'job_seek': None,
        'work_week_hrs': None,
        'learn_code': None,
        'learn_code_online': None,
        'coding_activities': None,
        'comp': 'Salary',
        'job_sat': 'JobSatisfaction',
        'language': 'HaveWorkedLanguage',
        'database': 'HaveWorkedDatabase',
        'platform': 'HaveWorkedPlatform',
        'webframe': None,
        'misc_tech': None,
        'op_sys_prof': None,
        'ai_use': None,
        'ai_sent': None
    },
    2018: {
        'file': 'survey_results_public2018.csv',
        'header': 0,
        'response_id': 'Respondent',
        'country': 'Country',
        'age': 'Age',
        'gender': 'Gender',
        'ethnicity': 'RaceEthnicity',
        'main_branch': None,
        'student': 'Student',
        'employment': 'Employment',
        'education': 'FormalEducation',
        'undergrad_major': 'UndergradMajor',
        'age_first_code': None,
        'org_size': 'CompanySize',
        'dev_type': 'DevType',
        'industry': None,
        'remote': None,
        'years_code': 'YearsCoding',
        'years_code_pro': 'YearsCodingProf',
        'work_exp': None,
        'job_seek': 'JobSearchStatus',
        'work_week_hrs': None,
        'learn_code': None,
        'learn_code_online': None,
        'coding_activities': None,
        'comp': 'ConvertedSalary',
        'job_sat': 'JobSatisfaction',
        'language': 'LanguageWorkedWith',
        'database': 'DatabaseWorkedWith',
        'platform': 'PlatformWorkedWith',
        'webframe': None,
        'misc_tech': None,
        'op_sys_prof': None,
        'ai_use': None,
        'ai_sent': 'AIFuture'
    },
    2019: {
        'file': 'survey_results_public2019.csv',
        'header': 0,
        'response_id': 'Respondent',
        'country': 'Country',
        'age': 'Age',
        'gender': 'Gender',
        'ethnicity': 'Ethnicity',
        'main_branch': 'MainBranch',
        'student': 'Student',
        'employment': 'Employment',
        'education': 'EdLevel',
        'undergrad_major': 'UndergradMajor',
        'age_first_code': 'Age1stCode',
        'org_size': 'OrgSize',
        'dev_type': 'DevType',
        'industry': None,
        'remote': 'WorkRemote',
        'years_code': 'YearsCode',
        'years_code_pro': 'YearsCodePro',
        'work_exp': None,
        'job_seek': 'JobSeek',
        'work_week_hrs': 'WorkWeekHrs',
        'learn_code': None,
        'learn_code_online': None,
        'coding_activities': None,
        'comp': 'ConvertedComp',
        'job_sat': 'JobSat',
        'language': 'LanguageWorkedWith',
        'database': 'DatabaseWorkedWith',
        'platform': 'PlatformWorkedWith',
        'webframe': 'WebFrameWorkedWith',
        'misc_tech': 'MiscTechWorkedWith',
        'op_sys_prof': None,
        'ai_use': None,
        'ai_sent': None
    },
    2020: {
        'file': 'survey_results_2020.csv',
        'header': 0,
        'response_id': 'Respondent',
        'country': 'Country',
        'age': 'Age',
        'gender': 'Gender',
        'ethnicity': 'Ethnicity',
        'main_branch': 'MainBranch',
        'student': None,
        'employment': 'Employment',
        'education': 'EdLevel',
        'undergrad_major': 'UndergradMajor',
        'age_first_code': 'Age1stCode',
        'org_size': 'OrgSize',
        'dev_type': 'DevType',
        'industry': None,
        'remote': None,
        'years_code': 'YearsCode',
        'years_code_pro': 'YearsCodePro',
        'work_exp': None,
        'job_seek': 'JobSeek',
        'work_week_hrs': 'WorkWeekHrs',
        'learn_code': None,
        'learn_code_online': None,
        'coding_activities': None,
        'comp': 'ConvertedComp',
        'job_sat': 'JobSat',
        'language': 'LanguageWorkedWith',
        'database': 'DatabaseWorkedWith',
        'platform': 'PlatformWorkedWith',
        'webframe': 'WebframeWorkedWith',
        'misc_tech': 'MiscTechWorkedWith',
        'op_sys_prof': None,
        'ai_use': None,
        'ai_sent': None
    },
    2021: {
        'file': 'survey_results_2021.csv',
        'header': 0,
        'response_id': 'ResponseId',
        'country': 'Country',
        'age': 'Age',
        'gender': 'Gender',
        'ethnicity': 'Ethnicity',
        'main_branch': 'MainBranch',
        'student': None,
        'employment': 'Employment',
        'education': 'EdLevel',
        'undergrad_major': None,
        'age_first_code': 'Age1stCode',
        'org_size': 'OrgSize',
        'dev_type': 'DevType',
        'industry': None,
        'remote': None,
        'years_code': 'YearsCode',
        'years_code_pro': 'YearsCodePro',
        'work_exp': None,
        'job_seek': None,
        'work_week_hrs': None,
        'learn_code': 'LearnCode',
        'learn_code_online': None,
        'coding_activities': None,
        'comp': 'ConvertedCompYearly',
        'job_sat': None,
        'language': 'LanguageHaveWorkedWith',
        'database': 'DatabaseHaveWorkedWith',
        'platform': 'PlatformHaveWorkedWith',
        'webframe': 'WebframeHaveWorkedWith',
        'misc_tech': 'MiscTechHaveWorkedWith',
        'op_sys_prof': None,
        'ai_use': None,
        'ai_sent': None
    },
    2022: {
        'file': 'survey_results_2022.csv',
        'header': 0,
        'response_id': 'ResponseId',
        'country': 'Country',
        'age': 'Age',
        'gender': 'Gender',
        'ethnicity': 'Ethnicity',
        'main_branch': 'MainBranch',
        'student': None,
        'employment': 'Employment',
        'education': 'EdLevel',
        'undergrad_major': None,
        'age_first_code': None,
        'org_size': 'OrgSize',
        'dev_type': 'DevType',
        'industry': None,
        'remote': 'RemoteWork',
        'years_code': 'YearsCode',
        'years_code_pro': 'YearsCodePro',
        'work_exp': 'WorkExp',
        'job_seek': None,
        'work_week_hrs': None,
        'learn_code': 'LearnCode',
        'learn_code_online': 'LearnCodeOnline',
        'coding_activities': 'CodingActivities',
        'comp': 'ConvertedCompYearly',
        'job_sat': None,
        'language': 'LanguageHaveWorkedWith',
        'database': 'DatabaseHaveWorkedWith',
        'platform': 'PlatformHaveWorkedWith',
        'webframe': 'WebframeHaveWorkedWith',
        'misc_tech': 'MiscTechHaveWorkedWith',
        'op_sys_prof': 'OpSysProfessional use',
        'ai_use': None,
        'ai_sent': None
    },
    2023: {
        'file': 'survey_results_2023.csv',
        'header': 0,
        'response_id': 'ResponseId',
        'country': 'Country',
        'age': 'Age',
        'gender': None,
        'ethnicity': None,
        'main_branch': 'MainBranch',
        'student': None,
        'employment': 'Employment',
        'education': 'EdLevel',
        'undergrad_major': None,
        'age_first_code': None,
        'org_size': 'OrgSize',
        'dev_type': 'DevType',
        'industry': 'Industry',
        'remote': 'RemoteWork',
        'years_code': 'YearsCode',
        'years_code_pro': 'YearsCodePro',
        'work_exp': 'WorkExp',
        'job_seek': None,
        'work_week_hrs': None,
        'learn_code': 'LearnCode',
        'learn_code_online': 'LearnCodeOnline',
        'coding_activities': 'CodingActivities',
        'comp': 'ConvertedCompYearly',
        'job_sat': None,
        'language': 'LanguageHaveWorkedWith',
        'database': 'DatabaseHaveWorkedWith',
        'platform': 'PlatformHaveWorkedWith',
        'webframe': 'WebframeHaveWorkedWith',
        'misc_tech': 'MiscTechHaveWorkedWith',
        'op_sys_prof': 'OpSysProfessional use',
        'ai_use': 'AISelect',
        'ai_sent': 'AISent'
    },
    2024: {
        'file': 'survey_results_2024.csv',
        'header': 0,
        'response_id': 'ResponseId',
        'country': 'Country',
        'age': 'Age',
        'gender': None,
        'ethnicity': None,
        'main_branch': 'MainBranch',
        'student': None,
        'employment': 'Employment',
        'education': 'EdLevel',
        'undergrad_major': None,
        'age_first_code': None,
        'org_size': 'OrgSize',
        'dev_type': 'DevType',
        'industry': 'Industry',
        'remote': 'RemoteWork',
        'years_code': 'YearsCode',
        'years_code_pro': 'YearsCodePro',
        'work_exp': 'WorkExp',
        'job_seek': None,
        'work_week_hrs': None,
        'learn_code': 'LearnCode',
        'learn_code_online': 'LearnCodeOnline',
        'coding_activities': 'CodingActivities',
        'comp': 'ConvertedCompYearly',
        'job_sat': 'JobSat',
        'language': 'LanguageHaveWorkedWith',
        'database': 'DatabaseHaveWorkedWith',
        'platform': 'PlatformHaveWorkedWith',
        'webframe': 'WebframeHaveWorkedWith',
        'misc_tech': 'MiscTechHaveWorkedWith',
        'op_sys_prof': 'OpSysProfessional use',
        'ai_use': 'AISelect',
        'ai_sent': 'AISent'
    },
    2025: {
        'file': 'survey_results_2025.csv',
        'header': 0,
        'response_id': 'ResponseId',
        'country': 'Country',
        'age': 'Age',
        'gender': None,
        'ethnicity': None,
        'main_branch': 'MainBranch',
        'student': None,
        'employment': 'Employment',
        'education': 'EdLevel',
        'undergrad_major': None,
        'age_first_code': None,
        'org_size': 'OrgSize',
        'dev_type': 'DevType',
        'industry': 'Industry',
        'remote': 'RemoteWork',
        'years_code': 'YearsCode',
        'years_code_pro': None,
        'work_exp': 'WorkExp',
        'job_seek': None,
        'work_week_hrs': None,
        'learn_code': 'LearnCode',
        'learn_code_online': None,
        'coding_activities': None,
        'comp': 'ConvertedCompYearly',
        'job_sat': 'JobSat',
        'language': 'LanguageHaveWorkedWith',
        'database': 'DatabaseHaveWorkedWith',
        'platform': 'PlatformHaveWorkedWith',
        'webframe': 'WebframeHaveWorkedWith',
        'misc_tech': None,
        'op_sys_prof': 'OpSysProfessional use',
        'ai_use': 'AISelect',
        'ai_sent': 'AISent'
    }
}

YEARS = sorted(YEAR_INFO)

# Consumer Price Index, use to account for inflation so salaries are adjusted properly
CPI_U = {
    2015: 237.017,
    2016: 240.007,
    2017: 245.120,
    2018: 251.107,
    2019: 255.657,
    2020: 258.811,
    2021: 270.970,
    2022: 292.655,
    2023: 304.702,
    2024: 313.689,
    2025: 321.943
}

# Mapping of the multiresponse fields separated into individual columns in 2015
EDU_2015_LEVELS = [
    ('No formal education / other', 'Training & Education: No formal training'),
    ('Some college/university', 'Training & Education: Some college, but no CS degree'),
    ("Bachelor's degree", 'Training & Education: BS in CS'),
    ("Master's degree", 'Training & Education: Masters in CS'),
    ('Doctoral degree', 'Training & Education: PhD in CS')
]

LANG_2015 = [
    'C',
    'C++',
    'C++11',
    'C#',
    'Clojure',
    'CoffeeScript',
    'Dart',
    'F#',
    'Go',
    'Haskell',
    'Java',
    'JavaScript',
    'Matlab',
    'Objective-C',
    'Perl',
    'PHP',
    'Python',
    'R',
    'Ruby',
    'Rust',
    'Scala',
    'SQL',
    'Swift',
    'Visual Basic'
]

DB_2015 = [
    'Cassandra',
    'MongoDB',
    'Redis',
    'SQL Server'
]

PLATFORM_2015 = [
    'Android',
    'AngularJS',
    'Arduino',
    'Cloud',
    'Cordova',
    'Hadoop',
    'iOS',
    'LAMP',
    'Node.js',
    'Salesforce',
    'Sharepoint',
    'Spark',
    'Windows Phone',
    'Wordpress'
]

TECH_2015_ALL = LANG_2015 + DB_2015 + PLATFORM_2015

# Country responses that cannot be mapped to a region or cleaned country name
COUNTRY_SPECIAL = {
    'Nomadic',
    'Other Country (Not Listed Above)',
    'I prefer not to say',
    'Other (please specify)'
}

# Weird country responses that have to be modified to be properly mapped by country_converter
COUNTRY_ALIAS = {
    'Venezuela, Bolivarian Republic of...': 'Venezuela',
    'Iran, Islamic Republic of...': 'Iran',
    'Viet Nam': 'Vietnam',
    'Republic of Korea': 'South Korea',
    'Korea South': 'South Korea',
    "Democratic People's Republic of Korea": 'North Korea',
    'Korea North': 'North Korea',
    'Hong Kong (S.A.R.)': 'Hong Kong',
    'Republic of Moldova': 'Moldova',
    'Moldavia': 'Moldova',
    'Bosnia Herzegovina': 'Bosnia and Herzegovina',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'The former Yugoslav Republic of Macedonia': 'North Macedonia',
    'Republic of North Macedonia': 'North Macedonia',
    'Macedonia': 'North Macedonia',
    "Lao People's Democratic Republic": 'Laos',
    'Syrian Arab Republic': 'Syria',
    'United Republic of Tanzania': 'Tanzania',
    'Libyan Arab Jamahiriya': 'Libya',
    'Myanmar, {Burma}': 'Myanmar',
    'Ireland {Republic}': 'Ireland',
    'Azerbaidjan': 'Azerbaijan',
    'Antigua & Deps': 'Antigua and Barbuda',
    'Trinidad & Tobago': 'Trinidad and Tobago',
    'Micronesia, Federated States of...': 'Micronesia',
    'Virgin Islands (USA)': 'United States Virgin Islands',
    'Reunion (French)': 'Reunion',
    'New Caledonia (French)': 'New Caledonia'
}

REGION_ALLOWED = {
    'Africa',
    'Americas',
    'Asia',
    'Europe',
    'Oceania'
}

cc = coco.CountryConverter()


# =====================================================================================
# Raw load helpers
# =====================================================================================

# Robust method to get a single midpoint value for many of the string range responses
def parse_midpoint(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    lower = text.lower()
    numbers = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', text.replace(',', ''))]

    if not numbers:
        return np.nan
    if len(numbers) >= 2:
        return sum(numbers[:2]) / 2
    if 'under' in lower or text.startswith('<'):
        return max(numbers[0] - 1, 0)
    return numbers[0]


# Creates a single DataFrame of separated 2015 multiselect columns
def join_selected(df, labels):
    cols = [f'Current Lang & Tech: {label}' for label in labels]
    flags = df.reindex(columns=cols).notna()
    parts = []
    for label in labels:
        col = f'Current Lang & Tech: {label}'
        parts.append(flags[col].map({True: label, False: pd.NA}))
    return pd.concat(parts, axis=1).apply(lambda row: ';'.join(row.dropna().astype(str)), axis=1).replace('', pd.NA)


# Builds one DataFrame for all 2015 separated multiselect options collapsed into one column each
def build_2015_fields(df):
    education = pd.Series(pd.NA, index=df.index, dtype='object')
    for label, col in EDU_2015_LEVELS:
        part = df[col].notna().map({True: label, False: pd.NA})
        education = education.fillna(part)

    language = join_selected(df, LANG_2015)
    database = join_selected(df, DB_2015)
    platform = join_selected(df, PLATFORM_2015)
    write_in = df.get(
        'Current Lang & Tech: Write-In',
        pd.Series(pd.NA, index=df.index)
    ).replace('', pd.NA)
    current_tech = pd.concat([join_selected(df, TECH_2015_ALL), write_in], axis=1).apply(
        lambda row: ';'.join(row.dropna().astype(str)),
        axis=1
    ).replace('', pd.NA)

    return pd.DataFrame({
        'education': education,
        'language': language,
        'database': database,
        'platform': platform,
        'current_tech': current_tech
    })


# Helper to quickly pull in the desired columns for a given year
def read_year(year, columns):
    info = YEAR_INFO[year]
    path = DATA_DIR / info['file']
    frame = pd.read_csv(
        path,
        header=info['header'],
        usecols=lambda col: col in columns,
        low_memory=False
    )
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        preview = ', '.join(missing[:10])
        raise ValueError(f"{path.name} is missing expected columns: {preview}")
    return frame


# Takes all the years and makes a unified DataFrame with a standardized format
def load_year(year):
    info = YEAR_INFO[year]
    selected = [info[field] for field in RAW_FIELDS if info[field] is not None]

    if year == 2015:
        selected += [col for _, col in EDU_2015_LEVELS]
        selected += [f'Current Lang & Tech: {label}' for label in TECH_2015_ALL]
        selected += ['Current Lang & Tech: Write-In']

    selected = list(dict.fromkeys(selected))
    df = read_year(year, selected)

    out = pd.DataFrame(index=df.index)
    out['row_id'] = year * 1_000_000 + np.arange(len(df))
    out['survey_year'] = year

    for field in RAW_FIELDS:
        raw_col = info[field]
        out[field] = df[raw_col] if raw_col in df.columns else pd.NA

    out['current_tech'] = pd.NA

    if year == 2015:
        derived = build_2015_fields(df)
        for col in derived.columns:
            out[col] = derived[col]

    out['age_num'] = out['age'].map(parse_midpoint)
    out['age_first_code_num'] = out['age_first_code'].map(parse_midpoint)
    out['years_code_num'] = out['years_code'].map(parse_midpoint)
    out['years_code_pro_num'] = out['years_code_pro'].map(parse_midpoint)
    out['work_exp_num'] = out['work_exp'].map(parse_midpoint)
    out['work_week_hrs_num'] = pd.to_numeric(out['work_week_hrs'], errors='coerce')
    out['comp_usd'] = out['comp'].map(parse_midpoint)
    out['job_sat_num'] = pd.to_numeric(out['job_sat'], errors='coerce')

    return out


# =====================================================================================
# Cleaning helpers
# =====================================================================================

# Don't want to type this for every Series with text fields I encounter
def lower_text(series):
    return series.fillna('').astype(str).str.lower().str.strip()


# Uses the country converter package to capture more countries and map to a region
def convert_one_country(name, target):
    value = cc.convert(names=name, to=target, not_found=None)
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0] if len(value) else pd.NA
    if value in {'not found', 'not found in regex', '', 'Other'}:
        return pd.NA
    return value


# Builds a fully cleaned DataFrame that maps countries to the region they are located in
def build_country_lookup(series):
    lookup = pd.DataFrame(
        {'country': sorted(series.dropna().astype(str).str.strip().unique())}
    )
    lookup['country_input'] = lookup['country'].replace(COUNTRY_ALIAS)
    lookup.loc[lookup['country'].isin(COUNTRY_SPECIAL), 'country_input'] = pd.NA
    lookup['country_clean'] = pd.NA
    lookup['region'] = pd.NA

    valid = lookup['country_input'].notna()
    lookup.loc[valid, 'country_clean'] = lookup.loc[valid, 'country_input'].map(
        lambda name: convert_one_country(name, 'name_short')
    )
    lookup.loc[valid, 'region'] = lookup.loc[valid, 'country_input'].map(
        lambda name: convert_one_country(name, 'continent')
    )
    lookup['country_clean'] = lookup['country_clean'].replace(['not found', 'not found in regex', '', 'Other'], pd.NA)
    lookup['region'] = lookup['region'].replace({
        'America': 'Americas',
        'Antarctica': pd.NA,
        'not found': pd.NA,
        'not found in regex': pd.NA,
        '': pd.NA,
        'Other': pd.NA
    })
    lookup['region'] = lookup['region'].where(lookup['region'].isin(REGION_ALLOWED), pd.NA)
    return lookup[['country', 'country_clean', 'region']]


# Education doesn't have standardized values across the years, this cleans inputs and groups them into appropriate categories
def clean_education(value):
    if pd.isna(value):
        return pd.NA
    text = re.sub(r'\s*\([^)]*\)', '', str(value).strip())
    lower = re.sub(r'master.?s', "master's", re.sub(r'bachelor.?s', "bachelor's", text.lower()))
    if 'prefer not' in lower:
        return pd.NA
    if 'never completed any formal education' in lower or 'self-taught' in lower or 'something else' in lower or 'no formal' in lower:
        return 'No formal education / other'
    if 'primary' in lower or 'elementary' in lower:
        return 'Primary/elementary school'
    if 'secondary school' in lower:
        return 'Secondary school'
    if 'some college' in lower or 'some university' in lower:
        return 'Some college/university'
    if 'associate degree' in lower:
        return 'Associate degree'
    if 'bachelor' in lower:
        return "Bachelor's degree"
    if 'master' in lower:
        return "Master's degree"
    if 'professional degree' in lower:
        return 'Professional degree'
    if 'doctoral degree' in lower or 'ph.d' in lower or 'ed.d' in lower:
        return 'Doctoral degree'
    return 'Other / ungrouped'


# Organization size also isn't standardized, with overlapping values and different text formatting. This fixes that
def clean_org_size(value):
    if pd.isna(value):
        return pd.NA
    lower = str(value).strip().lower()
    if any(term in lower for term in ["don't know", 'don?t know', 'not sure', 'prefer not']):
        return pd.NA
    if 'just me' in lower or 'not part of a company' in lower:
        return 'Self-employed'
    if any(term in lower for term in [
        '1-4 employees',
        '2 to 9 employees',
        '2-9 employees',
        '5-9 employees',
        'fewer than 10 employees',
        '10 to 19 employees',
        '10-19 employees',
        'less than 20 employees'
        ]
    ):
        return '1-19'
    if '20 to 99 employees' in lower or '20-99 employees' in lower:
        return '20-99'
    if any(term in lower for term in [
        '100 to 499 employees',
        '100-499 employees',
        '500 to 999 employees',
        '500-999 employees'
        ]
    ):
        return '100-999'
    if any(term in lower for term in [
        '1,000 to 4,999 employees',
        '1,000-4,999 employees',
        '5,000 to 9,999 employees',
        '5,000-9,999 employees'
        ]
    ):
        return '1,000-9,999'
    if '10,000 or more employees' in lower or '10,000+ employees' in lower:
        return '10,000+'
    return pd.NA


# Remote has many values that can vaguely describe the same thing, this creates three neat groups
def group_remote(value):
    if pd.isna(value):
        return pd.NA
    mostly_in_person = {
        'In-person',
        'Full in-person',
        'Never',
        'Less than once per month / Never',
        'I rarely work remote',
        'I rarely work remotely',
        'A few days each month'
    }
    hybrid = {
        'Less than half the time, but at least one day each week',
        'About half the time',
        'Part-time remote',
        'Part-time Remote',
        'Hybrid (some remote, leans heavy to in-person)',
        'Hybrid (some remote, some in-person)',
        'Your choice (very flexible, you can come in when you want or just as needed)'
    }
    mostly_remote = {
        'More than half, but not all, the time',
        'Hybrid (some in-person, leans heavy to flexibility)',
        'Fully remote',
        'Full-time remote',
        'Full-time Remote',
        "All or almost all the time (I'm full-time remote)",
        'Remote'
    }
    if value in mostly_in_person:
        return 'Mostly in-person'
    if value in hybrid:
        return 'Hybrid'
    if value in mostly_remote:
        return 'Mostly remote'
    return pd.NA


# Gets counts of levels for fields with multiple responses
def multi_count(series):
    counts = series.fillna('').astype(str).str.split(';').map(lambda items: sum(item.strip() not in {'', 'nan'} for item in items))
    return counts.where(series.notna(), np.nan)


# Creates potential role groups for the various responses across the dataset
def role_family(token):
    if pd.isna(token):
        return pd.NA
    text = str(token).strip().lower()
    if 'full-stack' in text:
        return 'Full-stack'
    if 'back-end' in text or 'backend' in text or 'server' in text:
        return 'Back-end'
    if 'front-end' in text or 'frontend' in text:
        return 'Front-end'
    if 'data' in text or 'machine learning' in text or 'scientist' in text or 'analyst' in text:
        return 'Data / ML'
    if 'mobile' in text or 'ios' in text or 'android' in text:
        return 'Mobile'
    if 'devops' in text or 'site reliability' in text or 'cloud' in text or 'system administrator' in text:
        return 'DevOps / Cloud'
    if 'desktop' in text or 'enterprise' in text:
        return 'Desktop / Enterprise'
    if 'manager' in text or 'executive' in text:
        return 'Management'
    if 'student' in text or 'academic researcher' in text or 'educator' in text:
        return 'Student / Academic'
    if 'qa' in text or 'quality assurance' in text or 'test' in text:
        return 'QA / Testing'
    return 'Other'


# =====================================================================================
# Main build
# =====================================================================================

def build_clean_core():
    clean = pd.concat([load_year(year) for year in YEARS], ignore_index=True)

    employment_text = lower_text(clean['employment'])
    branch_text = lower_text(clean['main_branch'])
    dev_text = lower_text(clean['dev_type'])
    student_text = lower_text(clean['student'])

    # Explicitly sets standard employment type groupings
    employment_group = np.select(
        [
            employment_text.str.contains('self-employed|independent contractor|freelance|contractor'),
            employment_text.str.contains('part-time') & employment_text.str.contains('employed'),
            employment_text.str.contains('full-time') & employment_text.str.contains('employed'),
            employment_text.str.contains('not employed'),
            student_text.str.contains('yes|student') | employment_text.str.contains('student')
        ],
        [
            'Independent / contract',
            'Employed part-time',
            'Employed full-time',
            'Not employed',
            'Student'
        ],
        default='Other'
    )

    clean['employment_group'] = pd.Series(employment_group, index=clean.index).where(
        clean['employment'].notna() | clean['student'].notna(),
        pd.NA
    )

    # Explicitly flags whether someone is employed or not
    clean['is_employed'] = (
        employment_text.str.contains('employed')
        | employment_text.str.contains('contractor')
        | employment_text.str.contains('freelance')
        | employment_text.str.contains('self-employed')
    ) & ~employment_text.str.contains('not employed')

    # Explicitly sets if someone works in data field professionally
    branch_prof = (
        branch_text.str.contains('developer by profession')
        | branch_text.eq('professional developer')
        | branch_text.eq('professional non-developer who sometimes writes code')
    )
    clean['is_professional'] = np.where(
        branch_text.ne(''),
        branch_prof,
        clean['is_employed'] & ~dev_text.str.contains('student')
    )

    # Final mapping of countries and regions
    country_lookup = build_country_lookup(clean['country'])
    clean = clean.drop(columns=['country_clean', 'region'], errors='ignore').merge(
        country_lookup,
        on='country',
        how='left'
    )

    # Sets each column to the standardized formats created earlier
    clean['education_clean'] = clean['education'].map(clean_education)
    clean['org_size_clean'] = clean['org_size'].map(clean_org_size)
    clean['remote_group'] = clean['remote'].map(group_remote)

    # Defines the ranges that we considered to be valid data to keep
    clean['age_mid'] = clean['age_num'].where(clean['age_num'].between(10, 90))
    clean['age_first_code_clean'] = clean['age_first_code_num'].where(clean['age_first_code_num'].between(5, 70))
    clean['years_code_clean'] = clean['years_code_num'].where(clean['years_code_num'].between(0, 50))
    clean['years_code_pro_clean'] = clean['years_code_pro_num'].where(clean['years_code_pro_num'].between(0, 50))
    clean['work_exp_clean'] = clean['work_exp_num'].where(clean['work_exp_num'].between(0, 50))
    clean['work_week_hrs_clean'] = clean['work_week_hrs_num'].where(clean['work_week_hrs_num'].between(1, 100))
    clean.loc[
        (clean['years_code_pro_clean'] > clean['years_code_clean'])
        & clean['years_code_clean'].notna(),
        'years_code_pro_clean'
    ] = np.nan
    clean['professional_experience_years'] = clean['years_code_pro_clean'].combine_first(clean['work_exp_clean'])
    clean['comp_usd_clean'] = clean['comp_usd'].where(clean['comp_usd'].between(1000, 1_000_000))
    clean['log_comp_usd_clean'] = np.log(clean['comp_usd_clean'])
    clean['age_group'] = pd.cut(
        clean['age_mid'],
        bins=[10, 25, 35, 45, 55, 100],
        labels=['Under 25', '25-34', '35-44', '45-54', '55+'],
        right=False
    )

    # Gets counts of levels for fields that have varied potential responses
    clean['language_count'] = multi_count(clean['language'])
    clean['database_count'] = multi_count(clean['database'])
    clean['platform_count'] = multi_count(clean['platform'])
    clean['webframe_count'] = multi_count(clean['webframe'])
    clean['misc_tech_count'] = multi_count(clean['misc_tech'])
    clean['learn_code_count'] = multi_count(clean['learn_code'])
    clean['learn_code_online_count'] = multi_count(clean['learn_code_online'])
    clean['coding_activities_count'] = multi_count(clean['coding_activities'])
    clean['op_sys_prof_count'] = multi_count(clean['op_sys_prof'])

    # Uses the Consumer Price Index to adjust compensation to all be level with 2025 inflation
    clean['cpi_u'] = clean['survey_year'].map(CPI_U)
    clean['comp_real_2025'] = clean['comp_usd_clean'] * CPI_U[2025] / clean['cpi_u']
    clean['log_comp_real_2025'] = np.log(clean['comp_real_2025'])

    # Defines responses likely to have usable compensation values
    clean['is_comp_analysis_sample'] = (
        clean['is_professional']
        & clean['comp_usd_clean'].notna()
        & clean['employment_group'].ne('Not employed')
    )

    # Defines model windows
    clean['is_comp_model_core'] = (clean['survey_year'] >= 2019) & clean['is_comp_analysis_sample']
    clean['is_comp_model_tech_rich'] = (clean['survey_year'] >= 2021) & clean['is_comp_analysis_sample']
    clean['is_comp_model_ai_era'] = (clean['survey_year'] >= 2023) & clean['is_comp_analysis_sample']
    clean['is_comp_model_extended'] = clean['is_comp_model_tech_rich']
    clean['is_comp_model_sample'] = clean['is_comp_model_core']

    # Takes the dev types and assigns them the role families we created
    roles_long = clean.loc[clean['dev_type'].notna(), ['row_id', 'dev_type']].copy()
    roles_long['dev_type'] = roles_long['dev_type'].astype(str).str.split(';')
    roles_long = roles_long.explode('dev_type')
    roles_long['dev_type'] = roles_long['dev_type'].str.strip()
    roles_long = roles_long.loc[roles_long['dev_type'].ne('')].copy()
    roles_long = roles_long.reset_index(drop=True)
    roles_long['role_family'] = roles_long['dev_type'].map(role_family)

    # Assigns given role family back to unique responses and standardizes their names
    role_flags = pd.crosstab(
        roles_long['row_id'].to_numpy(),
        roles_long['role_family'].to_numpy()
    ).clip(upper=1)
    role_flags.index.name = 'row_id'
    role_flags.columns = [
        'role_' + re.sub(r'[^a-z0-9]+', '_', col.lower()).strip('_')
        for col in role_flags.columns
    ]

    # Adds the assigned roles to the main DataFrame and gets their counts for later share calcs
    clean = clean.drop(columns=[col for col in clean.columns if col.startswith('role_')], errors='ignore')
    clean = clean.drop(columns=['role_family_count'], errors='ignore')
    clean = clean.merge(role_flags.reset_index(), on='row_id', how='left')
    role_cols = sorted([col for col in clean.columns if col.startswith('role_')])
    clean[role_cols] = clean[role_cols].fillna(0).astype(int)
    clean['role_family_count'] = clean[role_cols].sum(axis=1)

    # All of the columns that we want to keep for analysis
    clean_cols = [
        'row_id',
        'survey_year',
        'response_id',
        'country',
        'country_clean',
        'region',
        'age',
        'age_mid',
        'age_group',
        'gender',
        'ethnicity',
        'main_branch',
        'student',
        'employment',
        'employment_group',
        'is_employed',
        'is_professional',
        'education',
        'education_clean',
        'undergrad_major',
        'age_first_code',
        'age_first_code_clean',
        'org_size',
        'org_size_clean',
        'dev_type',
        'industry',
        'remote',
        'remote_group',
        'job_seek',
        'work_week_hrs',
        'work_week_hrs_clean',
        'learn_code',
        'learn_code_online',
        'coding_activities',
        'language',
        'database',
        'platform',
        'webframe',
        'misc_tech',
        'op_sys_prof',
        'current_tech',
        'ai_use',
        'ai_sent',
        'language_count',
        'database_count',
        'platform_count',
        'webframe_count',
        'misc_tech_count',
        'learn_code_count',
        'learn_code_online_count',
        'coding_activities_count',
        'op_sys_prof_count',
        'years_code',
        'years_code_clean',
        'years_code_pro',
        'work_exp',
        'professional_experience_years',
        'job_sat',
        'job_sat_num',
        'comp',
        'comp_usd_clean',
        'log_comp_usd_clean',
        'cpi_u',
        'comp_real_2025',
        'log_comp_real_2025',
        'is_comp_analysis_sample',
        'is_comp_model_core',
        'is_comp_model_tech_rich',
        'is_comp_model_ai_era',
        'is_comp_model_extended',
        'is_comp_model_sample',
        'role_family_count'
    ] + role_cols
    clean_core = clean[clean_cols].copy()

    text_cols = clean_core.select_dtypes(include=['object', 'string']).columns
    clean_core[text_cols] = clean_core[text_cols].astype('string')

    year_count = clean_core['survey_year'].nunique()
    if year_count != len(YEARS):
        raise ValueError(f'Expected {len(YEARS)} survey years but found {year_count}')

    return clean_core


# Takes the cleaned core DataFrame and writes it to a parquet file in data/derived/
def write_clean_core(path=OUT_PATH):
    clean_core = build_clean_core()
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    clean_core.to_parquet(path, index=False)
    return clean_core


# Makes everything run and outputs full path to write location
def main():
    clean_core = write_clean_core()
    print(str(OUT_PATH.resolve()))
    print(clean_core.shape)


# Runs file and lets us import its functions later without worrying about it auto running
if __name__ == '__main__':
    main()
