"""CLI entry point for the bert_SSI surveillance pipeline.

Usage:
    python scripts/run_pipeline.py --input data/notes.csv --output results/
    python scripts/run_pipeline.py --input data/pedw.csv --output results/ --mode structured_only
"""
import argparse
import os
from datetime import date
import pandas as pd
from src.pipeline.run import SSIPipeline
from src.output.formatter import filter_mdt_review
from src.output.summary import generate_summary


def main():
    parser = argparse.ArgumentParser(description="bert_SSI: Automated SSI surveillance pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--mode", choices=["auto", "text_only", "structured_only"], default="auto")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")

    print(f"Loading pipeline from {args.config}...")
    pipeline = SSIPipeline.from_config(args.config)
    if args.mode != "auto":
        pipeline.config["processing_mode"] = args.mode

    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input, dtype=str)
    print(f"Processing {len(df):,} episodes...")
    results = pipeline.run(df)

    results.to_csv(os.path.join(args.output, f"ssi_linelist_{stamp}.csv"), index=False)
    filter_mdt_review(results).to_csv(os.path.join(args.output, f"ssi_review_{stamp}.csv"), index=False)
    summary = generate_summary(results, date.today().isoformat(), pipeline.thresholds)
    with open(os.path.join(args.output, f"ssi_summary_{stamp}.txt"), "w") as f:
        f.write(summary)

    print(summary)
    print(f"\nOutputs written to {args.output}/")


if __name__ == "__main__":
    main()
