# Raw financial data

This folder contains public Apple Inc. SEC filings and financial-workbook exports used by the ingestion pipeline.

## Files

- `10K-2023.pdf`, `10K-2024.pdf`, `10K-2025.pdf`: annual Form 10-K filings.
- `10Q-2026.pdf`: quarterly Form 10-Q filing for the period ended June 27, 2026.
- Corresponding `.xls` files: structured filing data supplied with the source documents.
- `apple_metrics_curated.xlsx`: a normalized workbook derived only from the local 10-K filings.
- `manifest.csv`: ingestion metadata for every source file.

## Curated workbook policy

`apple_metrics_curated.xlsx` has two sheets. `metrics` is `public_financial`; `restricted_headcount` must be ingested with sensitivity `headcount_compensation`, irrespective of the workbook's manifest default. Every numeric value lists its filing and section in the `source` column. Financial values are expressed in USD, not millions.

The restrictive sheet is intentionally public-source data classified as restricted by this application's RBAC policy, so access control can be demonstrated without inventing data.
