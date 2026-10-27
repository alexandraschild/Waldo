"""Score the interpolation-scale sweep in results_interpolation_new/.

main_interpolation.py resizes each image by a multiplicative factor (the
"added_scales" values, e.g. 0.5 = half size, 1 = untouched, 5 = 5x upscale)
before feeding it to the model - see apply_image_transform in
src/benchmarks.py. Unlike the field/resized/bicubic sweep in
parse_field_results.py, scale=1 here already *is* the untransformed
baseline, so no separate results/results_old lookup is needed.

results_interpolation_new/{model}/{benchmark}/{benchmark}_{model_short}_{scale}_{timestamp}.json
holds one file per (model, benchmark, scale) sweep point; a run may leave
behind a `_partial.json` (interrupted) alongside or instead of the full file.

This reuses the rule-based scoring from parse_results.py but keeps one row
per (model, benchmark, scale) rather than collapsing scales together.

Usage:
    python -m src.parse_interpolation_results
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.parse_results import score_example


def model_short_name(model_dir_name: str) -> str:
    """Recover the short model name (e.g. "Qwen2.5-VL-3B-Instruct") from a
    results_interpolation_new/ directory name (e.g.
    "Qwen_Qwen2.5-VL-3B-Instruct"), which is the "org/repo" model name with
    "/" replaced by "_" (see run_interpolation_scripts.sh)."""
    return model_dir_name.split("_", 1)[1] if "_" in model_dir_name else model_dir_name


def find_scale_files(benchmark_dir: Path, benchmark: str, model_short: str) -> dict[float, tuple[Path, bool]]:
    """Map scale -> (path, is_partial) for every scale found in benchmark_dir,
    preferring a full file over a partial one and the latest timestamp among
    ties."""
    pattern = re.compile(
        rf"^{re.escape(benchmark)}_{re.escape(model_short)}_"
        rf"(?P<scale>[0-9]+(?:\.[0-9]+)?)_(?P<timestamp>\d{{8}}_\d{{6}})(?P<partial>_partial)?\.json$"
    )
    best: dict[float, tuple[Path, bool, str]] = {}
    for path in sorted(benchmark_dir.glob("*.json")):
        m = pattern.match(path.name)
        if not m:
            continue
        scale = float(m.group("scale"))
        is_partial = bool(m.group("partial"))
        timestamp = m.group("timestamp")
        # Rank full files above partial ones, then later timestamps above earlier.
        rank = (not is_partial, timestamp)

        current = best.get(scale)
        if current is None or rank > (not current[1], current[2]):
            best[scale] = (path, is_partial, timestamp)

    return {scale: (path, is_partial) for scale, (path, is_partial, _) in best.items()}


def score_interpolation_results(results_dir: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        model_short = model_short_name(model_dir.name)
        for benchmark_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            benchmark = benchmark_dir.name
            scale_files = find_scale_files(benchmark_dir, benchmark, model_short)

            for scale, (path, is_partial) in sorted(scale_files.items()):
                with open(path) as f:
                    examples = json.load(f)

                scored = []
                for ex in examples:
                    score, eval_method = score_example(benchmark, ex)
                    scored.append({**ex, "score": score, "eval_method": eval_method})

                out_path = out_dir / model_dir.name / benchmark / f"{benchmark}_{model_short}_{scale}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(scored, f, indent=2)

                n = len(scored)
                n_error = sum(1 for ex in scored if ex["eval_method"] == "model_error")
                n_unscorable = sum(1 for ex in scored if ex["score"] is None)
                scorable = [ex["score"] for ex in scored if ex["score"] is not None]
                accuracy = sum(scorable) / len(scorable) if scorable else float("nan")

                rows.append({
                    "model": model_dir.name,
                    "benchmark": benchmark,
                    "scale": scale,
                    "n": n,
                    "accuracy": accuracy,
                    "error_rate": n_error / n if n else float("nan"),
                    "unscorable_rate": n_unscorable / n if n else float("nan"),
                    "partial": is_partial,
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_interpolation_new"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_scored_interpolation"))
    args = parser.parse_args()

    df = score_interpolation_results(args.results_dir, args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "interpolation_summary.csv"
    df.sort_values(["model", "benchmark", "scale"]).to_csv(summary_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    for benchmark in sorted(df["benchmark"].unique()):
        bdf = df[df["benchmark"] == benchmark]
        print(f"=== {benchmark}: accuracy by model x scale ===")
        print((bdf.pivot(index="model", columns="scale", values="accuracy") * 100).round(1))
        print()

    print("=== Mean accuracy across benchmarks, by model x scale ===")
    print((df.pivot_table(index="model", columns="scale", values="accuracy", aggfunc="mean") * 100).round(1))

    print("\n=== Mean accuracy across scales, by model (ranked) ===")
    print((df.groupby("model")["accuracy"].mean() * 100).round(1).sort_values(ascending=False))

    err = (df.groupby("model")["error_rate"].mean() * 100).round(1).sort_values(ascending=False)
    if (err > 0).any():
        print("\n=== Model error rate (raw generation errors, e.g. crashes) ===")
        print(err[err > 0])

    partial = df[df["partial"]]
    if not partial.empty:
        print("\n=== Scored from _partial.json (run incomplete) ===")
        print(partial[["model", "benchmark", "scale", "n"]].to_string(index=False))

    print(f"\nWrote per-example scored results to {args.out_dir}/ and summary to {summary_path}")


if __name__ == "__main__":
    main()
