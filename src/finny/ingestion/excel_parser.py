"""Excel ingestor for curated workbooks and SEC .xls financial statements."""

from pathlib import Path
from typing import Any, Dict, List
import hashlib
import pandas as pd
import openpyxl
import xlrd


class ExcelIngestor:
    """Parses .xlsx and .xls financial data into normalized JSON records with sensitivity tags."""

    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = raw_data_dir

    def process_file(self, file_path: Path, manifest_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Processes an Excel file based on manifest info and filename conventions."""
        file_name = file_path.name

        if "curated" in file_name.lower():
            return self._parse_curated_workbook(file_path, manifest_info)
        else:
            return self._parse_sec_xls(file_path, manifest_info)

    def _parse_curated_workbook(self, file_path: Path, manifest_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parses apple_metrics_curated.xlsx.
        
        Sheet 'metrics' inherits default_sensitivity (public_financial).
        Sheet 'restricted_headcount' is force-tagged with 'headcount_compensation'.
        """
        records: List[Dict[str, Any]] = []
        excel_file = pd.ExcelFile(file_path)

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            if df.empty:
                continue

            # Determine sensitivity tag based on policy
            if sheet_name == "restricted_headcount":
                sensitivity = "headcount_compensation"
            else:
                sensitivity = manifest_info.get("default_sensitivity", "public_financial")

            # Clean column names
            df.columns = [str(c).strip().lower() for c in df.columns]

            for idx, row in df.iterrows():
                metric_name = str(row.get("metric", "")).strip()
                if not metric_name or metric_name.lower() == "nan":
                    continue

                period = str(row.get("period", manifest_info.get("fiscal_year", ""))).strip()
                val = row.get("value", None)
                unit = str(row.get("unit", "USD")).strip() if pd.notna(row.get("unit")) else "USD"
                source = str(row.get("source", f"{manifest_info.get('doc_type', '10-K')} {period}")).strip()

                # Parse float/int value
                try:
                    if pd.notna(val):
                        numeric_val = float(val)
                    else:
                        continue
                except (ValueError, TypeError):
                    continue

                rec_id = hashlib.sha256(
                    f"{file_path.name}_{sheet_name}_{idx}_{metric_name}_{period}".encode("utf-8")
                ).hexdigest()[:16]

                record = {
                    "id": f"metric_{rec_id}",
                    "type": "financial_metric",
                    "metric": metric_name,
                    "period": period,
                    "value": numeric_val,
                    "unit": unit,
                    "source": source,
                    "sensitivity": sensitivity,
                    "file": file_path.name,
                    "sheet": sheet_name,
                    "doc_type": manifest_info.get("doc_type", "curated_metrics"),
                    "fiscal_year": manifest_info.get("fiscal_year", "2023-2025"),
                }
                records.append(record)

        return records

    def _parse_sec_xls(self, file_path: Path, manifest_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parses SEC .xls financial statement tables into metric records."""
        records: List[Dict[str, Any]] = []
        try:
            excel_file = pd.ExcelFile(file_path, engine="xlrd")
        except Exception:
            return records

        default_sens = manifest_info.get("default_sensitivity", "public_financial")
        doc_type = manifest_info.get("doc_type", "10-K_financial_data")
        fiscal_year = str(manifest_info.get("fiscal_year", ""))

        for sheet_name in excel_file.sheet_names:
            if sheet_name.lower() in ["table_of_contents", "subsidiaries"]:
                continue
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            if df.empty or df.shape[1] < 2:
                continue

            # Locate year header row
            year_row_idx = None
            col_years = {}
            for r in range(0, min(35, len(df))):
                row_items = [str(x).strip() for x in df.iloc[r].values if pd.notna(x)]
                # Skip EDGAR metadata rows
                if any(v.startswith(("Created by", "Form Type", "Period End:", "Date Filed:")) for v in row_items):
                    continue
                year_matches = [v for v in row_items if any(y in v for y in ["2020", "2021", "2022", "2023", "2024", "2025", "2026"])]
                if len(year_matches) >= 1:
                    year_row_idx = r
                    for c_idx in range(df.shape[1]):
                        cell_v = str(df.iloc[r, c_idx]).replace("\n", " ").strip()
                        if pd.notna(df.iloc[r, c_idx]) and any(y in cell_v for y in ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]):
                            col_years[c_idx] = cell_v
                    break

            if year_row_idx is None or not col_years:
                continue

            # Process data rows below header
            for r_idx in range(year_row_idx + 1, len(df)):
                row = df.iloc[r_idx]

                # Collect non-numeric metric label text parts from row
                text_parts = []
                for c_idx in range(df.shape[1]):
                    val = row.iloc[c_idx]
                    if pd.notna(val):
                        v_str = str(val).strip()
                        clean_v = v_str.replace(",", "").replace("$", "").replace("(", "").replace(")", "").strip()
                        try:
                            float(clean_v)
                        except ValueError:
                            if len(v_str) > 1 and not v_str.startswith(("Created by", "Form Type", "Period End", "Date Filed", "Table Of Contents")):
                                text_parts.append(v_str)

                metric_label = " ".join(text_parts).strip()
                if not metric_label or len(metric_label) < 3:
                    continue

                for c_idx, period_label in col_years.items():
                    val = row.iloc[c_idx]
                    if pd.isna(val):
                        continue

                    val_str = str(val).replace(",", "").replace("$", "").strip()
                    if not val_str or val_str in ["-", "—"]:
                        continue

                    is_neg = False
                    if val_str.startswith("(") and val_str.endswith(")"):
                        is_neg = True
                        val_str = val_str[1:-1].strip()

                    try:
                        numeric_val = float(val_str)
                        if is_neg:
                            numeric_val = -numeric_val
                    except ValueError:
                        continue

                    rec_id = hashlib.sha256(
                        f"{file_path.name}_{sheet_name}_{r_idx}_{c_idx}_{metric_label}".encode("utf-8")
                    ).hexdigest()[:16]

                    record = {
                        "id": f"metric_{rec_id}",
                        "type": "financial_metric",
                        "metric": metric_label,
                        "period": period_label,
                        "value": numeric_val,
                        "unit": "USD (in millions)",
                        "source": f"{file_path.name} sheet '{sheet_name}'",
                        "sensitivity": default_sens,
                        "file": file_path.name,
                        "sheet": sheet_name,
                        "doc_type": doc_type,
                        "fiscal_year": fiscal_year,
                    }
                    records.append(record)

        return records
