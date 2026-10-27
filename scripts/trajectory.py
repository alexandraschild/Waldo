#!/usr/bin/env python3
"""
Build combined binary correctness DataFrames for VLM trajectory analysis.

Structure:
  Baseline: results0/<model_id>/<benchmark>/<benchmark>_<model>_0_<timestamp>.json
  Experiments: evaluated1/<experiment>/<model_id>/<benchmark>/<benchmark>_<model>_<scale>_<timestamp>.json

Each JSON file is a list of:
  {"id": ..., "question": ..., "generated_answer": ..., "ground_truth": ...,
   "input_tokens": [text_tokens, image_tokens], "score": 1.0 or 0.0, ...}

Output: One CSV per model with rows=image_ids, columns=experiment_scale, values=0/1
"""

# Resolve shared project code and default outputs independently of the invoking directory.
import sys as _sys
from pathlib import Path as _Path
_REPOSITORY_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))
if __name__ == "__main__":
    import os as _os
    _os.chdir(_REPOSITORY_ROOT)


import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional


# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENTS = ['1d_center', '1d_topleft', '2d_center', '2d_topleft', 'interpolation']
SCALES = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0]


def parse_scale_from_filename(filename: str) -> Optional[float]:
    """
    Extract scale from filename: {benchmark}_{model}_{scale}_{timestamp}.json
    Strategy: split by '_', find the part that is a valid scale number.
    The scale is typically the second-to-last segment before timestamp.
    """
    stem = Path(filename).stem  # remove .json
    parts = stem.split('_')

    # Walk from right: timestamp is last (digits), scale is before that
    # But timestamp could also be numeric, so we check against known scales
    all_known = [0, 0.0] + SCALES

    # Try from right side
    for i in range(len(parts) - 1, -1, -1):
        try:
            val = float(parts[i])
            if val in all_known:
                return val
        except ValueError:
            continue

    return None


def load_json_file(filepath: str) -> list:
    """Load a JSON file containing a list of result records."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'results' in data:
        return data['results']
    elif isinstance(data, dict) and 'data' in data:
        return data['data']
    else:
        return [data]


def process_single_file(filepath: str, experiment: str, scale: float) -> list:
    """Process one JSON file and return list of (image_id, column_name, score) tuples."""
    records = load_json_file(filepath)
    results = []

    # Column name: e.g., "baseline_0" or "1d_center_1.5"
    col_name = f"{experiment}_{scale}"

    for record in records:
        image_id = record.get('id', record.get('image_id', ''))
        score = record.get('score', None)

        if score is None:
            # Fallback: compute from answers
            gen = str(record.get('generated_answer', '')).strip().lower()
            gt = str(record.get('ground_truth', '')).strip().lower()
            score = 1.0 if gen == gt else 0.0

        results.append({
            'image_id': image_id,
            'column': col_name,
            'score': int(score),
        })

    return results


def build_model_dataframe(results0_dir: str, evaluated1_dir: str,
                          model_id: str, benchmark: str) -> pd.DataFrame:
    """
    Build combined DataFrame for one model.
    Rows = image_ids, Columns = experiment_scale, Values = 0/1
    """
    all_records = []

    # --- Load baseline ---
    baseline_path = Path(results0_dir) / model_id / benchmark
    if baseline_path.exists():
        json_files = list(baseline_path.glob("*.json"))
        print(f"  Baseline: {len(json_files)} files in {baseline_path}")
        for jf in json_files:
            records = process_single_file(str(jf), 'baseline', 0)
            all_records.extend(records)
    else:
        print(f"  WARNING: Baseline path not found: {baseline_path}")

    # --- Load experiments ---
    for experiment in EXPERIMENTS:
        exp_path = Path(evaluated1_dir) / experiment / model_id / benchmark

        if not exp_path.exists():
            print(f"  WARNING: Not found: {exp_path}")
            continue

        json_files = list(exp_path.glob("*.json"))
        print(f"  {experiment}: {len(json_files)} files")

        for jf in json_files:
            scale = parse_scale_from_filename(jf.name)
            if scale is None:
                print(f"    WARNING: Cannot parse scale from {jf.name}")
                continue
            # Skip scale=0 in experiment folders (that's baseline)
            if scale == 0:
                continue

            records = process_single_file(str(jf), experiment, scale)
            all_records.extend(records)

    if not all_records:
        print(f"  ERROR: No records loaded for {model_id}")
        return pd.DataFrame()

    # --- Pivot to wide format ---
    long_df = pd.DataFrame(all_records)
    print(f"  Total records: {len(long_df)}")
    print(f"  Unique images: {long_df['image_id'].nunique()}")
    print(f"  Unique columns: {long_df['column'].nunique()}")

    # Pivot: rows=image_id, columns=experiment_scale, values=score
    wide_df = long_df.pivot_table(
        index='image_id',
        columns='column',
        values='score',
        aggfunc='first'
    )

    # Sort columns logically
    col_order = ['baseline_0']
    for exp in EXPERIMENTS:
        for scale in SCALES:
            col_name = f"{exp}_{scale}"
            if col_name in wide_df.columns:
                col_order.append(col_name)

    # Add any remaining columns not in our expected order
    remaining = [c for c in wide_df.columns if c not in col_order]
    col_order = [c for c in col_order if c in wide_df.columns] + remaining

    wide_df = wide_df[col_order]

    print(f"  Final shape: {wide_df.shape}")
    print(f"  Columns: {list(wide_df.columns)}")

    return wide_df


def main():
    parser = argparse.ArgumentParser(
        description='Build combined correctness DataFrames per model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/trajectory.py --results0 ./results0 --evaluated1 ./evaluated1 \\
      --models Qwen2-VL-7B Qwen2.5-VL-7B --benchmark hr_bench --output_dir ./dataframes
        """
    )
    parser.add_argument('--results0', type=str, default='./results0',
                        help='Baseline results directory')
    parser.add_argument('--evaluated1', type=str, default='./evaluated1',
                        help='Experiment results directory')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='Model IDs (folder names)')
    parser.add_argument('--benchmark', type=str, required=True,
                        help='Benchmark name (folder name)')
    parser.add_argument('--output_dir', type=str, default='./dataframes',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for model_id in args.models:
        print(f"\n{'='*60}")
        print(f"Processing: {model_id}")
        print(f"{'='*60}")

        df = build_model_dataframe(args.results0, args.evaluated1, model_id, args.benchmark)

        if len(df) > 0:
            output_path = os.path.join(args.output_dir, f"{model_id}_{args.benchmark}.csv")
            df.to_csv(output_path)
            print(f"\n  Saved: {output_path}")

            # Quick summary
            print(f"\n  Summary:")
            print(f"    Baseline accuracy: {df['baseline_0'].mean():.3f}" if 'baseline_0' in df.columns else "    No baseline")
            for exp in EXPERIMENTS:
                exp_cols = [c for c in df.columns if c.startswith(exp)]
                if exp_cols:
                    exp_acc = df[exp_cols].mean().mean()
                    print(f"    {exp:15s} avg accuracy: {exp_acc:.3f} ({len(exp_cols)} scales)")
        else:
            print(f"\n  SKIPPED: No data for {model_id}")

    print("\nDone!")


if __name__ == '__main__':
    main()