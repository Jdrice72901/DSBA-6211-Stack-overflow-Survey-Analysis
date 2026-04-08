# Data Storage

This folder stores local Stack Overflow Annual Developer Survey data extracts and derived analysis artifacts.

Raw files:

- Local CSV extracts currently cover survey years 2011-2025
- Treat raw annual CSVs as immutable source data from <https://survey.stackoverflow.co/>
- Do not overwrite raw files with cleaned, filtered, or harmonized versions

Derived files:

- `derived/clean_core.parquet` is the current harmonized respondent-year table built by `src/comp_clean.py`
- The current derived table covers 2015-2025; 2011-2014 remain inventory-only until separate survey-semantics validation is completed
- Any new derived data should preserve raw fields where practical and document source years, row counts, filters, and harmonization rules
