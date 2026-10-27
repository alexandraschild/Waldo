"""Score the LLaVA field-experiment triplet: field / resized / bicubic.

Per src/utils.py:build_field_images, each field value produces three image
variants sharing the same field=0 (untransformed) baseline:
  * field   - original embedded in a larger white-bordered canvas,
  * resized - original plain image downscaled by the coefficient the field
              image experiences during preprocessing (isolates content
              downscaling from the added-border effect),
  * bicubic - `resized` scaled back up to the original resolution via
              bicubic interpolation (detail loss without a size change).

results/llava-hf_llava-1.5-7b-hf/{benchmark}/{benchmark}_llava-1.5-7b-hf_{variant}{v}.json
holds the v=0.5..5 sweep for each variant. v=0 (no transformation at all, so
identical across all three variants) predates the field/resized/bicubic
naming split and was never rerun, so it still lives under
results_old/llava-hf_llava-1.5-7b-hf/{benchmark}/{benchmark}_llava-1.5-7b-hf_0.json.

This reuses the rule-based scoring from parse_results.py but keeps one row per
(variant, benchmark, field_value) instead of collapsing all variants into
duplicate model/benchmark rows the way results_scored/summary.csv does, so the
effect of each transformation - and how it compares to the other two - is
visible.

Usage:
    python -m src.parse_field_results
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.parse_results import score_example

MODEL = "llava-hf_llava-1.5-7b-hf"
MODEL_SHORT = "llava-1.5-7b-hf"
FIELD_VALUES = ["0", "0.5", "1", "2", "3", "4", "5"]
VARIANTS = ["field", "resized", "bicubic"]


def find_variant_file(new_dir: Path, old_dir: Path, benchmark: str, variant: str, value: str,
                       model_short: str = MODEL_SHORT) -> tuple[Path | None, bool]:
    """Return (path, is_partial) for a given variant/field value, or (None, False)."""
    if value == "0":
        # field=0 is untransformed, so field/resized/bicubic share one baseline file.
        path = old_dir / benchmark / f"{benchmark}_{model_short}_0.json"
        return (path, False) if path.exists() else (None, False)

    path = new_dir / benchmark / f"{benchmark}_{model_short}_{variant}{value}.json"
    if path.exists():
        return path, False
    partial = new_dir / benchmark / f"{benchmark}_{model_short}_{variant}{value}_partial.json"
    if partial.exists():
        return partial, True
    return None, False


def score_field_results(results_dir: Path, results_old_dir: Path, out_dir: Path,
                         model: str = MODEL, model_short: str = MODEL_SHORT) -> pd.DataFrame:
    """Score field/resized/bicubic results for ``model`` (a ``results/<model>``-style
    directory name, e.g. ``"Qwen_Qwen2-VL-7B-Instruct"``), defaulting to LLaVA-1.5-7B
    for backwards compatibility with existing callers."""
    new_dir = results_dir / model
    old_dir = results_old_dir / model
    benchmarks = sorted(p.name for p in new_dir.iterdir() if p.is_dir())

    rows = []
    for benchmark in benchmarks:
        for variant in VARIANTS:
            for value in FIELD_VALUES:
                path, is_partial = find_variant_file(new_dir, old_dir, benchmark, variant, value, model_short)
                if path is None:
                    rows.append({
                        "variant": variant, "benchmark": benchmark, "field_value": float(value), "n": 0,
                        "accuracy": float("nan"), "error_rate": float("nan"),
                        "unscorable_rate": float("nan"), "partial": False,
                    })
                    continue

                with open(path) as f:
                    examples = json.load(f)

                scored = []
                for ex in examples:
                    score, eval_method = score_example(benchmark, ex)
                    scored.append({**ex, "score": score, "eval_method": eval_method})

                out_path = out_dir / model / variant / benchmark / f"{benchmark}_{model_short}_{variant}{value}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(scored, f, indent=2)

                n = len(scored)
                n_error = sum(1 for ex in scored if ex["eval_method"] == "model_error")
                n_unscorable = sum(1 for ex in scored if ex["score"] is None)
                scorable = [ex["score"] for ex in scored if ex["score"] is not None]
                accuracy = sum(scorable) / len(scorable) if scorable else float("nan")

                rows.append({
                    "variant": variant,
                    "benchmark": benchmark,
                    "field_value": float(value),
                    "n": n,
                    "accuracy": accuracy,
                    "error_rate": n_error / n if n else float("nan"),
                    "unscorable_rate": n_unscorable / n if n else float("nan"),
                    "partial": is_partial,
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--results-old-dir", type=Path, default=Path("results_old"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_scored_field"))
    args = parser.parse_args()

    df = score_field_results(args.results_dir, args.results_old_dir, args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "field_summary.csv"
    df.sort_values(["variant", "benchmark", "field_value"]).to_csv(summary_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    for variant in VARIANTS:
        vdf = df[df["variant"] == variant]
        print(f"=== {MODEL}: {variant} accuracy by benchmark x field value ===")
        print((vdf.pivot(index="benchmark", columns="field_value", values="accuracy") * 100).round(1))
        print()

    print("=== Mean accuracy across benchmarks, by variant x field value ===")
    print((df.pivot_table(index="variant", columns="field_value", values="accuracy", aggfunc="mean") * 100).round(1))

    missing = df[df["n"] == 0]
    if not missing.empty:
        print("\n=== Missing (no result file found) ===")
        print(missing[["variant", "benchmark", "field_value"]].to_string(index=False))

    partial = df[df["partial"]]
    if not partial.empty:
        print("\n=== Scored from _partial.json (run incomplete) ===")
        print(partial[["variant", "benchmark", "field_value", "n"]].to_string(index=False))

    print(f"\nWrote per-example scored results to {args.out_dir}/ and summary to {summary_path}")


if __name__ == "__main__":
    main()
