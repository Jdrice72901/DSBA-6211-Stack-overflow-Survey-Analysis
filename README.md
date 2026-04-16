# DSBA 6211 Stack Overflow Survey Analysis

This repository supports a DSBA 6211 project analysis of Stack Overflow Annual Developer Survey data from 2011-2025. The project focuses on survey-aware analysis of global respondent patterns, regional differences, compensation, job satisfaction, employment duration, and subgroup stories within the Stack Overflow community.

Official survey source: <https://survey.stackoverflow.co/>

Current workflow:

- Raw annual CSV extracts live under `data/` and should be treated as immutable source files
- The main standardized respondent-year table is built by `src/comp_clean.py`
- The current production derived table covers 2015-2025 and is written to `data/derived/clean_core.parquet`
- `clean_core.parquet` is intentionally lean for general use whereas model-specific features and flags are surfaced in their respective notebooks or pipeline steps
- Raw 2011-2014 files are present, but are inventory-only as their questionnaire semantics are different and offer less coverage
- Compensation modeling currently uses a 2019+ window because compensation fields are more comparable in later years
- Job satisfaction modeling requires explicit year-specific target standardization as satisfaction questions change across survey years

Interpretation caveats:

- Survey years are repeated cross-sections, not a panel of the same respondents
- Stack Overflow survey respondents are a convenience sample from the Stack Overflow community, not all developers worldwide
- Questionnaire wording, skip logic, response options, and denominator definitions change across years
- Model results should be interpreted as descriptive, predictive, or associational only
