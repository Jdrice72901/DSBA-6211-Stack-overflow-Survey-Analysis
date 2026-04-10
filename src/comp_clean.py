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
    'employment_addl',
    'education',
    'undergrad_major',
    'age_first_code',
    'org_size',
    'ic_or_pm',
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
        'employment_addl': None,
        'education': 'EdLevel',
        'undergrad_major': None,
        'age_first_code': None,
        'org_size': 'OrgSize',
        'ic_or_pm': 'ICorPM',
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
        'employment_addl': None,
        'education': 'EdLevel',
        'undergrad_major': None,
        'age_first_code': None,
        'org_size': 'OrgSize',
        'ic_or_pm': 'ICorPM',
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
        'employment_addl': None,
        'education': 'EdLevel',
        'undergrad_major': None,
        'age_first_code': None,
        'org_size': 'OrgSize',
        'ic_or_pm': 'ICorPM',
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
        'employment_addl': 'EmploymentAddl',
        'education': 'EdLevel',
        'undergrad_major': None,
        'age_first_code': None,
        'org_size': 'OrgSize',
        'ic_or_pm': 'ICorPM',
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

INVENTORY_ONLY_INFO = {
    2011: {
        'file': '2011 Stack Overflow Survey Results.csv',
        'header': 0
    },
    2012: {
        'file': '2012 Stack Overflow Survey Results.csv',
        'header': 0
    },
    2013: {
        'file': '2013 Stack Overflow Survey Responses.csv',
        'header': 0
    },
    2014: {
        'file': '2014 Stack Overflow Survey Responses.csv',
        'header': 0
    }
}

NUMERIC_RULES = [
    {
        'field': 'age',
        'clean_field': 'age_mid',
        'min': 10,
        'max': 90
    },
    {
        'field': 'age_first_code',
        'clean_field': 'age_first_code_clean',
        'min': 5,
        'max': 70
    },
    {
        'field': 'years_code',
        'clean_field': 'years_code_clean',
        'min': 0,
        'max': 50
    },
    {
        'field': 'years_code_pro',
        'clean_field': 'years_code_pro_clean',
        'min': 0,
        'max': 50
    },
    {
        'field': 'work_exp',
        'clean_field': 'work_exp_clean',
        'min': 0,
        'max': 50
    },
    {
        'field': 'work_week_hrs',
        'clean_field': 'work_week_hrs_clean',
        'min': 1,
        'max': 100
    },
    {
        'field': 'comp',
        'clean_field': 'comp_usd_clean',
        'min': 1_000,
        'max': 1_000_000
    }
]

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

# 2016 put all tech into a single column, these separate them into their respective ones
DB_2016 = DB_2015.copy()
LANG_2016 = LANG_2015.copy()

PLAT_2016 = [
    'Android',
    'Arduino / Raspberry Pi',
    'Cloud (AWS, GAE, Azure, etc.)',
    'Cordova',
    'Hadoop',
    'iOS',
    'LAMP',
    'Node.js',
    'Salesforce',
    'SharePoint',
    'Spark',
    'Windows Phone',
    'WordPress'
]

# These are techs that aren't really a language, platform, or database and are thus dropped
EXTRAS = [
    'AngularJS',
    'APT',
    'Ansible',
    'Bun',
    'Cargo',
    'Chocolatey',
    'Composer',
    'Datadog',
    'Zephyr',
    'Flow',
    'Google Cloud Storage',
    'Gradle',
    'Homebrew',
    'MSBuild',
    'Make',
    'Maven (build tool)',
    'Microsoft Azure (Tables, CosmosDB, SQL, etc)',
    'New Relic',
    'Ninja',
    'NuGet',
    'Other(s):',
    'Pacman',
    'Pip',
    'Poetry',
    'Prometheus',
    'ReactJS',
    'Slack',
    'Slack Apps and Integrations',
    'Splunk',
    'Terraform',
    'Vite',
    'Webpack',
    'Yarn',
    'npm',
    'pnpm'
]

# Used to correctly determine what to keep and remove for 2016 tech columns
DB_REMOVE = set(LANG_2016 + PLAT_2016 + EXTRAS)
LANG_REMOVE = set(DB_2016 + PLAT_2016 + EXTRAS)
PLAT_REMOVE = set(DB_2016 + LANG_2016 + EXTRAS)

# Standardizes language responses across the years
LANG_MAP = {
    'C++11': 'C++',
    'Matlab': 'MATLAB',
    'Cobol': 'COBOL',
    'Ocaml': 'OCaml',
    'Delphi': 'Delphi/Object Pascal',
    'LISP': 'Lisp',
    'Common Lisp': 'Lisp',
    'Bash/Shell': 'Bash/Shell/PowerShell',
    'Bash/Shell (all shells)': 'Bash/Shell/PowerShell',
    'PowerShell': 'Bash/Shell/PowerShell',
    'HTML': 'HTML/CSS',
    'CSS': 'HTML/CSS',
    'Visual Basic (.Net)': 'VB.NET',
    'Visual Basic 6': 'Visual Basic'
}

# Standardizes platform responses across the years
PLAT_MAP = {
    'Amazon Web Services (AWS)': 'AWS',
    'Arduino': 'Arduino / Raspberry Pi',
    'Raspberry Pi': 'Arduino / Raspberry Pi',
    'Azure': 'Microsoft Azure',
    'Cloud': 'Generic Cloud',
    'Cloud (AWS, GAE, Azure, etc.)': 'Generic Cloud',
    'Google Cloud Platform/App Engine': 'Google Cloud',
    'Google Cloud Platform': 'Google Cloud',
    'IBM Cloud or Watson': 'IBM Cloud',
    'IBM Cloud Or Watson': 'IBM Cloud',
    'Digital Ocean': 'DigitalOcean',
    'Oracle Cloud Infrastructure (OCI)': 'Oracle Cloud Infrastructure',
    'Linode, now Akamai': 'Linode',
    'Mac OS': 'macOS',
    'MacOS': 'macOS',
    'Linux Desktop': 'Linux',
    'Windows Desktop': 'Windows',
    'Windows Desktop or Server': 'Windows',
    'Sharepoint': 'SharePoint',
    'Wordpress': 'WordPress'
}

# Standardizes database responses across the years
DB_MAP = {
    'SQL Server': 'Microsoft SQL Server',
    'Amazon DynamoDB': 'DynamoDB',
    'Dynamodb': 'DynamoDB',
    'IBM Db2': 'IBM DB2',
    'Google BigQuery': 'BigQuery',
    'Couch DB': 'CouchDB',
    'Neo4J': 'Neo4j',
    'Firebase': 'Firebase Realtime Database'
}

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
    if 'under' in lower or 'less than' in lower or 'younger than' in lower or text.startswith('<'):
        return max(numbers[0] / 2, 0)
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

# Used to correctly attribute mixed multiselect columns to the correct single column
def clean_multiselect(value, keep=None, remove=None, replace=None):
    if pd.isna(value):
        return pd.NA

    values = []
    seen = set()
    keep = set(keep) if keep is not None else None
    replace = replace or {}
    remove = remove or set()

    for item in str(value).split(';'):
        token = item.strip()
        if token in {'', 'nan'}:
            continue
        if keep is not None and token not in keep:
            continue
        if token in remove:
            continue
        token = replace.get(token, token)
        if token in remove or token in seen:
            continue
        values.append(token)
        seen.add(token)

    return ';'.join(values) if values else pd.NA


# 2016 has its tech fields grouped together, this properly separates them
def build_2016_fields(df):
    tech = df['tech_do']
    return pd.DataFrame({
        'language': tech.map(
            lambda value: clean_multiselect(value, keep=LANG_2016, remove=LANG_REMOVE, replace=LANG_MAP)
        ),
        'database': tech.map(
            lambda value: clean_multiselect(value, keep=DB_2016, remove=DB_REMOVE, replace=DB_MAP)
        ),
        'platform': tech.map(
            lambda value: clean_multiselect(value, keep=PLAT_2016, remove=PLAT_REMOVE, replace=PLAT_MAP)
        ),
        'current_tech': tech.map(clean_multiselect)
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
    selected = [info.get(field) for field in RAW_FIELDS if info.get(field) is not None]

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
        raw_col = info.get(field)
        out[field] = df[raw_col] if raw_col in df.columns else pd.NA

    out['current_tech'] = pd.NA

    if year == 2015:
        derived = build_2015_fields(df)
        for col in derived.columns:
            out[col] = derived[col]
    if year == 2016:
        derived = build_2016_fields(df)
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
    if any(term in lower for term in ["don't know", 'don?t know', 'don’t know', 'not sure', 'prefer not']):
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


# Cleans later-wave contributor vs manager signal into one stable field
def clean_ic_or_pm(value):
    if pd.isna(value):
        return pd.NA
    lower = str(value).strip().lower()
    if 'contributor' in lower:
        return 'Individual contributor'
    if 'manager' in lower:
        return 'People manager'
    if 'prefer not' in lower:
        return pd.NA
    return pd.NA


# Industry gets messy across years, so this keeps the buckets broad and stable
def clean_industry(value, survey_year):
    if pd.isna(value) or survey_year == 2017:
        return pd.NA

    lower = str(value).strip().lower()
    lower = lower.replace('&', 'and')

    if any(term in lower for term in ['prefer not', "i'm a student", 'not currently employed']):
        return pd.NA
    if lower in {'other', 'other:', 'other (please specify)'}:
        return 'Other / unclear'
    if any(term in lower for term in [
        'software development',
        'software products',
        'web services',
        'information services',
        'technology',
        'internet',
        'telecomm',
        'telecommunications',
        'computer systems design'
    ]):
        return 'Software / IT'
    if any(term in lower for term in [
        'financial services',
        'finance / banking',
        'banking/financial services',
        'fintech',
        'insurance'
    ]):
        return 'Financial services'
    if 'healthcare' in lower:
        return 'Healthcare'
    if any(term in lower for term in [
        'manufacturing',
        'transportation',
        'supply chain',
        'automotive',
        'aerospace',
        'defense'
    ]):
        return 'Manufacturing / logistics'
    if any(term in lower for term in ['government', 'public sector', 'state-owned']):
        return 'Government / public'
    if 'education' in lower or 'university' in lower or 'school' in lower:
        return 'Education'
    if any(term in lower for term in ['retail', 'consumer products', 'consumer services', 'wholesale']):
        return 'Retail / consumer'
    if any(term in lower for term in ['media', 'advertising']):
        return 'Media / advertising'
    if any(term in lower for term in ['energy', 'oil and gas', 'oil & gas', 'utilities']):
        return 'Energy / utilities'
    if any(term in lower for term in ['consulting', 'legal services']):
        return 'Professional services'
    if any(term in lower for term in ['non-profit', 'nonprofit', 'foundation']):
        return 'Non-profit'
    return 'Other / unclear'


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
    if 'architect' in text:
        return 'Architecture'
    if 'developer advocate' in text or 'developer experience' in text or 'devrel' in text:
        return 'Advocacy / DX'
    if 'designer' in text or 'ux' in text or 'ui' in text:
        return 'Design / UX'
    if 'embedded' in text or 'firmware' in text or 'hardware' in text:
        return 'Embedded / hardware'
    if 'game' in text or 'graphics' in text:
        return 'Game / graphics'
    if 'full-stack' in text:
        return 'Full-stack'
    if 'back-end' in text or 'backend' in text or 'server' in text:
        return 'Back-end'
    if 'front-end' in text or 'frontend' in text:
        return 'Front-end'
    if 'web developer' in text:
        return 'Full-stack'
    if 'data' in text or 'machine learning' in text or 'scientist' in text or 'analyst' in text or 'ai/ml' in text:
        return 'Data / ML'
    if 'mobile' in text or 'ios' in text or 'android' in text:
        return 'Mobile'
    if any(term in text for term in ['devops', 'site reliability', 'cloud', 'system administrator', 'systems administrator', 'sysadmin']):
        return 'DevOps / Cloud'
    if 'desktop' in text or 'enterprise' in text:
        return 'Desktop / Enterprise'
    if 'security' in text:
        return 'Security'
    if 'manager' in text or 'executive' in text:
        return 'Management'
    if 'student' in text or 'academic researcher' in text or 'educator' in text:
        return 'Student / Academic'
    if 'qa' in text or 'quality assurance' in text or 'test' in text:
        return 'QA / Testing'
    return 'Other'


# =====================================================================================
# Audit and validation helpers
# =====================================================================================

def load_clean_core(path=OUT_PATH):
    return pd.read_parquet(path)


def survey_file_inventory():
    rows = []
    all_info = {
        **INVENTORY_ONLY_INFO,
        **{
            year: {
                'file': info['file'],
                'header': info['header']
            }
            for year, info in YEAR_INFO.items()
        }
    }

    for year, info in sorted(all_info.items()):
        path = DATA_DIR / info['file']
        if not path.exists():
            rows.append({
                'survey_year': year,
                'file': info['file'],
                'status': 'missing',
                'analysis_status': 'inventory_only' if year in INVENTORY_ONLY_INFO else 'clean_core',
                'row_count': np.nan,
                'column_count': np.nan
            })
            continue

        columns = pd.read_csv(
            path,
            header=info['header'],
            nrows=0,
            encoding_errors='ignore'
        ).columns
        rows.append({
            'survey_year': year,
            'file': info['file'],
            'status': 'present',
            'analysis_status': 'inventory_only' if year in INVENTORY_ONLY_INFO else 'clean_core',
            'row_count': sum(1 for _ in path.open(encoding='utf-8', errors='ignore')) - info['header'] - 1,
            'column_count': len(columns)
        })

    return pd.DataFrame(rows)


def _parsed_numeric(clean_core, field):
    if field == 'work_week_hrs':
        return pd.to_numeric(clean_core[field], errors='coerce')
    return clean_core[field].map(parse_midpoint)


def audit_numeric_masks(clean_core):
    rows = []

    for rule in NUMERIC_RULES:
        field = rule['field']
        parsed = _parsed_numeric(clean_core, field)
        in_range = parsed.between(rule['min'], rule['max'])
        clean_field = rule['clean_field']
        clean_non_null = clean_core[clean_field].notna().sum() if clean_field in clean_core.columns else np.nan

        rows.append({
            'field': field,
            'clean_field': clean_field or 'not_retained_directly',
            'min_valid': rule['min'],
            'max_valid': rule['max'],
            'source_non_null': int(parsed.notna().sum()),
            'range_masked': int((parsed.notna() & ~in_range).sum()),
            'clean_non_null': clean_non_null
        })

    return pd.DataFrame(rows)


def audit_country_region(clean_core):
    return (
        clean_core
        .groupby('survey_year')
        .agg(
            rows=('row_id', 'size'),
            country_non_null=('country', lambda series: int(series.notna().sum())),
            country_clean_missing=('country_clean', lambda series: int(series.isna().sum())),
            region_missing=('region', lambda series: int(series.isna().sum()))
        )
        .reset_index()
    )


def audit_unmapped_countries(clean_core, sample_col=None):
    frame = clean_core.copy()
    if sample_col is not None:
        frame = frame.loc[frame[sample_col]].copy()

    unmapped = frame.loc[
        frame['country'].notna() & frame['country_clean'].isna(),
        ['survey_year', 'country']
    ].copy()
    unmapped['country_input'] = unmapped['country'].replace(COUNTRY_ALIAS)
    unmapped['is_special_case'] = unmapped['country'].isin(COUNTRY_SPECIAL)

    return (
        unmapped
        .groupby(['survey_year', 'country', 'country_input', 'is_special_case'])
        .size()
        .rename('rows')
        .reset_index()
        .sort_values(['survey_year', 'rows', 'country'], ascending=[True, False, True])
    )


def audit_comp_outliers(clean_core):
    parsed = clean_core['comp'].map(parse_midpoint)
    frame = clean_core.assign(comp_parsed=parsed)
    return (
        frame
        .groupby('survey_year')
        .agg(
            comp_raw_non_null=('comp_parsed', lambda series: int(series.notna().sum())),
            comp_below_floor=('comp_parsed', lambda series: int(series.lt(1_000).sum())),
            comp_above_cap=('comp_parsed', lambda series: int(series.gt(1_000_000).sum())),
            comp_exact_cap=('comp_parsed', lambda series: int(series.eq(1_000_000).sum())),
            comp_clean_non_null=('comp_usd_clean', lambda series: int(series.notna().sum())),
            comp_model_rows=('is_comp_model_sample', lambda series: int(series.sum()))
        )
        .reset_index()
    )


def audit_clean_core(clean_core=None):
    clean_core = load_clean_core() if clean_core is None else clean_core
    return {
        'year_counts': clean_core.groupby('survey_year').size().rename('rows').reset_index(),
        'country_region': audit_country_region(clean_core),
        'unmapped_countries': audit_unmapped_countries(clean_core),
        'numeric_masks': audit_numeric_masks(clean_core),
        'comp_outliers': audit_comp_outliers(clean_core)
    }


def validate_clean_core(clean_core=None):
    clean_core = load_clean_core() if clean_core is None else clean_core
    errors = []

    required = {
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
        'comp_usd_clean',
        'log_comp_real_2025',
        'comp_real_2025',
        'is_comp_analysis_sample',
        'is_comp_model_core',
        'is_comp_model_tech_rich',
        'is_comp_model_ai_era',
        'is_comp_model_sample'
    }
    missing = sorted(required - set(clean_core.columns))
    if missing:
        errors.append(f"Missing required derived columns: {', '.join(missing)}")

    if clean_core['survey_year'].nunique() != len(YEARS):
        errors.append(f"Expected {len(YEARS)} survey years but found {clean_core['survey_year'].nunique()}")
    if set(clean_core['survey_year'].dropna().unique()) != set(YEARS):
        errors.append('Derived survey years do not match YEAR_INFO keys')
    if not clean_core['row_id'].is_unique:
        errors.append('row_id must be unique')

    response_ids = clean_core.loc[clean_core['response_id'].notna(), ['survey_year', 'response_id']]
    if response_ids.duplicated().any():
        errors.append('response_id must be unique within survey_year when present')

    region_values = set(clean_core['region'].dropna().unique())
    if not region_values.issubset(REGION_ALLOWED):
        extra = ', '.join(sorted(region_values - REGION_ALLOWED))
        errors.append(f"Unexpected region values: {extra}")

    if not clean_core['comp_usd_clean'].dropna().between(1_000, 1_000_000).all():
        errors.append('comp_usd_clean must be between 1,000 and 1,000,000 when present')

    log_rows = clean_core['log_comp_real_2025'].notna()
    if log_rows.any():
        expected_log = np.log(clean_core.loc[log_rows, 'comp_real_2025'])
        if not np.allclose(clean_core.loc[log_rows, 'log_comp_real_2025'], expected_log):
            errors.append('log_comp_real_2025 must equal log(comp_real_2025)')

    if not clean_core.loc[clean_core['country_clean'].isna(), 'region'].isna().all():
        errors.append('region should be missing when country_clean is missing')

    if not clean_core.loc[
        clean_core['years_code_pro_clean'].notna() & clean_core['years_code_clean'].notna(),
        'years_code_pro_clean'
    ].le(
        clean_core.loc[
            clean_core['years_code_pro_clean'].notna() & clean_core['years_code_clean'].notna(),
            'years_code_clean'
        ]
    ).all():
        errors.append('years_code_pro_clean must be less than or equal to years_code_clean when both are present')

    employment_allowed = {
        'Employed full-time',
        'Employed part-time',
        'Independent / contract',
        'Student',
        'Retired',
        'Not employed',
        'Other'
    }
    employment_values = set(clean_core['employment_primary'].dropna().unique())
    if not employment_values.issubset(employment_allowed):
        extra = ', '.join(sorted(employment_values - employment_allowed))
        errors.append(f"Unexpected employment_primary values: {extra}")

    ic_or_pm_allowed = {
        'Individual contributor',
        'People manager'
    }
    ic_or_pm_values = set(clean_core['ic_or_pm_clean'].dropna().unique())
    if not ic_or_pm_values.issubset(ic_or_pm_allowed):
        extra = ', '.join(sorted(ic_or_pm_values - ic_or_pm_allowed))
        errors.append(f"Unexpected ic_or_pm_clean values: {extra}")

    if not (clean_core['is_comp_model_core'] == (clean_core['is_comp_analysis_sample'] & clean_core['survey_year'].ge(2019))).all():
        errors.append('is_comp_model_core must match 2019+ compensation analysis sample')
    if not (clean_core['is_comp_model_tech_rich'] == (clean_core['is_comp_analysis_sample'] & clean_core['survey_year'].ge(2021))).all():
        errors.append('is_comp_model_tech_rich must match 2021+ compensation analysis sample')
    if not (clean_core['is_comp_model_ai_era'] == (clean_core['is_comp_analysis_sample'] & clean_core['survey_year'].ge(2023))).all():
        errors.append('is_comp_model_ai_era must match 2023+ compensation analysis sample')
    if not (clean_core['is_comp_model_sample'] == clean_core['is_comp_model_core']).all():
        errors.append('is_comp_model_sample must currently alias is_comp_model_core')

    if errors:
        raise ValueError('\n'.join(errors))

    return True


# =====================================================================================
# Main build
# =====================================================================================

def build_clean_core():
    clean = pd.concat([load_year(year) for year in YEARS], ignore_index=True)

    employment_text = lower_text(clean['employment'])
    employment_addl_text = lower_text(clean['employment_addl'])
    branch_text = lower_text(clean['main_branch'])
    dev_text = lower_text(clean['dev_type'])
    student_text = lower_text(clean['student'])

    # 2025 split employment into a main status and add-on statuses, so we keep both the anchor and overlap
    addl_part_time = employment_addl_text.str.contains(
        'engaged in paid work \\(less than 10 hours per week\\)|engaged in paid work \\(10-19 hours per week\\)|engaged in paid work \\(20-29 hours per week\\)'
    )
    addl_student = employment_addl_text.str.contains('attending school')
    addl_retired = employment_addl_text.str.contains('transitioning to retirement')

    clean['is_independent'] = employment_text.str.contains('self-employed|independent contractor|freelance|contractor')
    clean['is_full_time_employed'] = employment_text.str.contains('full-time') | (employment_text.eq('employed') & ~addl_part_time)
    clean['is_part_time_employed'] = (employment_text.str.contains('part-time') | addl_part_time) & ~clean['is_full_time_employed']
    clean['is_student_status'] = student_text.str.contains('yes|student') | employment_text.str.contains('student') | addl_student
    clean['is_retired_status'] = employment_text.str.contains('retired') | addl_retired
    clean['is_not_employed'] = employment_text.str.contains('not employed') & ~(
        clean['is_full_time_employed']
        | clean['is_part_time_employed']
        | clean['is_independent']
    )

    employment_primary = np.select(
        [
            employment_text.eq('employed') & addl_part_time,
            employment_text.eq('employed'),
            employment_text.eq('independent contractor, freelancer, or self-employed'),
            employment_text.eq('student'),
            employment_text.eq('retired'),
            employment_text.eq('not employed'),
            clean['is_full_time_employed'],
            clean['is_independent'],
            clean['is_part_time_employed'],
            clean['is_student_status'],
            clean['is_retired_status'],
            clean['is_not_employed']
        ],
        [
            'Employed part-time',
            'Employed full-time',
            'Independent / contract',
            'Student',
            'Retired',
            'Not employed',
            'Employed full-time',
            'Independent / contract',
            'Employed part-time',
            'Student',
            'Retired',
            'Not employed'
        ],
        default='Other'
    )

    clean['employment_primary'] = pd.Series(employment_primary, index=clean.index).where(
        clean['employment'].notna() | clean['student'].notna() | clean['employment_addl'].notna(),
        pd.NA
    )
    clean['employment_group'] = clean['employment_primary']
    clean['is_paid_worker'] = clean['is_full_time_employed'] | clean['is_part_time_employed'] | clean['is_independent']
    clean['is_employed'] = clean['is_paid_worker']

    # Explicitly sets if someone works in the field professionally
    branch_prof = (
        branch_text.str.contains('developer by profession')
        | branch_text.eq('professional developer')
        | branch_text.eq('professional non-developer who sometimes writes code')
    )
    clean['is_professional'] = np.where(
        branch_text.ne(''),
        branch_prof,
        clean['is_paid_worker'] & ~dev_text.str.contains('student')
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
    clean['industry_clean'] = [
        clean_industry(value, year)
        for value, year in zip(clean['industry'], clean['survey_year'], strict=False)
    ]
    clean['ic_or_pm_clean'] = clean['ic_or_pm'].map(clean_ic_or_pm)

    # Clean multiselect text so counts are based on real values rather than raw survey quirks
    clean['language'] = clean['language'].map(
        lambda value: clean_multiselect(value, remove=LANG_REMOVE, replace=LANG_MAP)
    )
    clean['database'] = clean['database'].map(
        lambda value: clean_multiselect(value, remove=DB_REMOVE, replace=DB_MAP)
    )
    clean['platform'] = clean['platform'].map(
        lambda value: clean_multiselect(value, remove=PLAT_REMOVE, replace=PLAT_MAP)
    )
    clean['webframe'] = clean['webframe'].map(clean_multiselect)
    clean['misc_tech'] = clean['misc_tech'].map(clean_multiselect)
    clean['op_sys_prof'] = clean['op_sys_prof'].map(clean_multiselect)
    clean['current_tech'] = clean['current_tech'].map(clean_multiselect)

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
    clean['experience_proxy_years'] = clean['years_code_pro_clean'].combine_first(clean['work_exp_clean'])
    clean['experience_proxy_source'] = np.select(
        [
            clean['years_code_pro_clean'].notna(),
            clean['work_exp_clean'].notna()
        ],
        [
            'years_code_pro',
            'work_exp'
        ],
        default=pd.NA
    )
    clean['professional_experience_years'] = clean['experience_proxy_years']
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
        & clean['is_paid_worker']
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
        'employment_addl',
        'employment_primary',
        'employment_group',
        'is_employed',
        'is_paid_worker',
        'is_full_time_employed',
        'is_part_time_employed',
        'is_independent',
        'is_student_status',
        'is_retired_status',
        'is_not_employed',
        'is_professional',
        'education',
        'education_clean',
        'undergrad_major',
        'age_first_code',
        'age_first_code_clean',
        'org_size',
        'org_size_clean',
        'ic_or_pm',
        'ic_or_pm_clean',
        'dev_type',
        'industry',
        'industry_clean',
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
        'years_code_pro_clean',
        'work_exp',
        'work_exp_clean',
        'experience_proxy_years',
        'experience_proxy_source',
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
