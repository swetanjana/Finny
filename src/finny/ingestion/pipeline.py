"""Pipeline orchestrator for ingesting raw financial PDFs and Excel files."""

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from rich.console import Console
from rich.table import Table

from finny.ingestion.excel_parser import ExcelIngestor
from finny.ingestion.pdf_parser import PDFIngestor

console = Console()


class IngestionPipeline:
    """Orchestrates end-to-end ingestion of raw filing files into normalized JSONL outputs."""

    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.manifest_path = raw_dir / "manifest.csv"
        self.excel_ingestor = ExcelIngestor(raw_dir)
        self.pdf_ingestor = PDFIngestor()

    def load_manifest(self) -> List[Dict[str, Any]]:
        """Loads metadata from manifest.csv."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found at {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        manifest_items = df.to_dict(orient="records")
        return manifest_items

    def run(self) -> Dict[str, Any]:
        """Runs full ingestion workflow over all raw files."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        manifest_items = self.load_manifest()
        manifest_by_file = {item["file"]: item for item in manifest_items}

        all_metrics: List[Dict[str, Any]] = []
        all_narratives: List[Dict[str, Any]] = []
        file_stats: Dict[str, Dict[str, Any]] = {}

        console.print("[bold blue]Starting Finny Ingestion Pipeline...[/bold blue]")

        # Process manifest files
        for item in manifest_items:
            file_name = item["file"]
            file_path = self.raw_dir / file_name

            if not file_path.exists():
                console.print(f"[bold red]Warning: File {file_name} in manifest does not exist at {file_path}[/bold red]")
                continue

            file_format = str(item.get("format", "")).lower()
            console.print(f"Processing [cyan]{file_name}[/cyan] ({file_format})...")

            if file_format == "pdf":
                chunks = self.pdf_ingestor.process_file(file_path, item)
                all_narratives.extend(chunks)
                file_stats[file_name] = {
                    "format": "pdf",
                    "doc_type": item.get("doc_type"),
                    "count": len(chunks),
                    "sensitivities": set(c["sensitivity"] for c in chunks),
                }
            elif file_format in ["xls", "xlsx"]:
                metrics = self.excel_ingestor.process_file(file_path, item)
                all_metrics.extend(metrics)
                file_stats[file_name] = {
                    "format": file_format,
                    "doc_type": item.get("doc_type"),
                    "count": len(metrics),
                    "sensitivities": set(m["sensitivity"] for m in metrics),
                }

        # Write output JSONL files
        metrics_file = self.processed_dir / "financial_metrics.jsonl"
        with open(metrics_file, "w", encoding="utf-8") as f:
            for m in all_metrics:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        narratives_file = self.processed_dir / "narrative_chunks.jsonl"
        with open(narratives_file, "w", encoding="utf-8") as f:
            for n in all_narratives:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")

        # Calculate summary statistics
        sensitivity_counts = {}
        for rec in all_metrics + all_narratives:
            sens = rec["sensitivity"]
            sensitivity_counts[sens] = sensitivity_counts.get(sens, 0) + 1

        summary = {
            "total_files_processed": len(file_stats),
            "total_financial_metrics": len(all_metrics),
            "total_narrative_chunks": len(all_narratives),
            "sensitivity_breakdown": sensitivity_counts,
            "output_files": {
                "financial_metrics": str(metrics_file.resolve()),
                "narrative_chunks": str(narratives_file.resolve()),
            },
        }

        summary_file = self.processed_dir / "ingestion_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._print_summary_table(summary, file_stats)

        return summary

    def _print_summary_table(self, summary: Dict[str, Any], file_stats: Dict[str, Dict[str, Any]]):
        """Displays summary tables in console using rich."""
        table = Table(title="Finny Ingestion Summary — File Level")
        table.add_column("File", style="cyan")
        table.add_column("Format", style="magenta")
        table.add_column("Doc Type", style="green")
        table.add_column("Records Extracted", justify="right", style="yellow")
        table.add_column("Sensitivity Tags", style="bold blue")

        for f_name, stats in file_stats.items():
            table.add_row(
                f_name,
                stats["format"],
                str(stats["doc_type"]),
                str(stats["count"]),
                ", ".join(sorted(list(stats["sensitivities"]))),
            )

        console.print(table)

        summary_table = Table(title="Sensitivity Taxonomy Breakdown")
        summary_table.add_column("Sensitivity Level", style="cyan")
        summary_table.add_column("Record Count", justify="right", style="green")

        for sens, cnt in summary["sensitivity_breakdown"].items():
            summary_table.add_row(sens, str(cnt))

        console.print(summary_table)
        console.print(f"[bold green]Ingestion complete! Outputs written to {self.processed_dir}[/bold green]")


def main():
    root_dir = Path(__file__).resolve().parents[3]
    raw_dir = root_dir / "data" / "raw"
    processed_dir = root_dir / "data" / "processed"
    pipeline = IngestionPipeline(raw_dir, processed_dir)
    pipeline.run()


if __name__ == "__main__":
    main()
