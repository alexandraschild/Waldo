"""Parse and score the raw generation results under results/.

The files in results/{model}/{benchmark}/{benchmark}_{model}_0.json contain raw
(unscored) model generations: no "score" field is present yet, and the
LLM-judge pipeline in evaluate.py depends on a "custom_parcing" module that
does not exist in this repo. This script fills that gap with cheap,
dependency-free rule-based scoring so results can be inspected without an
OpenAI key, and writes per-example scored copies plus an accuracy summary.

Usage:
    python -m src.parse_results
    python -m src.parse_results --results-dir results --out-dir results_scored
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Benchmarks whose ground truth is a multiple-choice letter (possibly
# parenthesised, e.g. "(A)") and whose generated_answer is free text that
# should contain that letter as its selected option.
MCQ_BENCHMARKS = {"MMStar", "HRBench", "CVBench", "VStarBench", "MMERealWorldLite"}

# Short-answer VQA-style benchmarks: ground truth is a short phrase/number,
# graded with normalized exact/contains match.
SHORT_ANSWER_BENCHMARKS = {"VQAv2", "DocVQA", "TextVQA", "ChartQA", "InfographicVQA", "GQA"}

# Free-form long-answer benchmarks. True grading needs an LLM judge; we fall
# back to substring containment as a rough (lower-bound) proxy and mark it as
# such via eval_method="heuristic_contains".
OPEN_ENDED_BENCHMARKS = {"MMVet", "CharXiv"}

# Yes/No benchmarks.
YESNO_BENCHMARKS = {"MMEval"}

# Mixed MCQ + free-form math benchmark: try letter match, then numeric/string
# containment.
MIXED_BENCHMARKS = {"MathVerse"}

LETTER_RE = re.compile(r"\(?\b([A-F])\b\)?[.:]?")
PUNCT_RE = re.compile(r"[^\w\s%.\-]")


def is_error(generated_answer: str) -> bool:
    return isinstance(generated_answer, str) and generated_answer.strip().upper().startswith("ERROR")


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = PUNCT_RE.sub("", text)
    return " ".join(text.split())


def extract_letter(text: str) -> str | None:
    """Best-effort extraction of a single A-F option letter from free text.

    Prefers the *last* standalone letter match, since models often reason
    first ("... so the answer is C") before stating the final choice, and
    also handles the case where the whole answer already is just "C" or
    "(C)".
    """
    stripped = text.strip()
    if len(stripped) <= 3:
        m = LETTER_RE.match(stripped.upper())
        if m:
            return m.group(1)
    matches = LETTER_RE.findall(text.upper())
    return matches[-1] if matches else None


OPTION_PAREN_RE = re.compile(r"\(([A-F])\)\s*([^\n]+)")
OPTION_INLINE_RE = re.compile(r"\s*([A-F])[:.]\s*(.+)")


def parse_options(question: str) -> dict[str, str]:
    """Recover the {letter: option_text} mapping embedded in an MCQ prompt.

    Handles the two formats seen in these benchmarks: "(A) text" repeated
    one-per-line (HRBench, CVBench, VStarBench, MMERealWorldLite), and
    "Options: A: text, B: text, ..." / "Choices:\nA:text\nB:text" (MMStar,
    MathVerse).
    """
    options = {m.group(1): m.group(2).strip() for m in OPTION_PAREN_RE.finditer(question)}
    if options:
        return options

    section = question
    for marker in ("Options:", "Choices:"):
        if marker in question:
            section = question.split(marker, 1)[1]
            break
    for fragment in re.split(r"[\n,]", section):
        m = OPTION_INLINE_RE.match(fragment)
        if m:
            options[m.group(1)] = m.group(2).strip()
    return options


def match_option_by_text(generated: str, options: dict[str, str]) -> str | None:
    """Fallback for answers that restate the option's text instead of its
    letter (e.g. "The umbrella is located below the car." for choice
    "(B) below"). Only returns a letter if exactly one option text matches,
    to avoid guessing under ambiguity.
    """
    gen_norm = normalize(generated)
    if not gen_norm:
        return None
    matches = {
        letter
        for letter, text in options.items()
        if normalize(text) and (normalize(text) in gen_norm or gen_norm in normalize(text))
    }
    return matches.pop() if len(matches) == 1 else None


def score_mcq(generated: str, ground_truth: str, question: str = "") -> float | None:
    gt_letter = extract_letter(ground_truth) or normalize(ground_truth).upper()
    gen_letter = extract_letter(generated)
    if gen_letter is None and question:
        gen_letter = match_option_by_text(generated, parse_options(question))
    if gen_letter is None:
        return None
    return 1.0 if gen_letter == gt_letter else 0.0


YESNO_RE = re.compile(r"\b(yes|no)\b")


def extract_yesno(text: str) -> str | None:
    stripped = text.strip().lower()
    if stripped.startswith("yes"):
        return "yes"
    if stripped.startswith("no"):
        return "no"
    matches = YESNO_RE.findall(stripped)
    return matches[-1] if matches else None


def score_yesno(generated: str, ground_truth: str) -> float | None:
    gen_answer = extract_yesno(generated)
    if gen_answer is None:
        return None
    return 1.0 if gen_answer == normalize(ground_truth) else 0.0


def score_short_answer(generated: str, ground_truth: str) -> float:
    gen_norm = normalize(generated)
    gt_norm = normalize(ground_truth)
    if not gt_norm:
        return 0.0
    return 1.0 if gen_norm == gt_norm or gt_norm in gen_norm else 0.0


def score_open_ended(generated: str, ground_truth: str) -> float:
    gt_norm = normalize(ground_truth)
    gen_norm = normalize(generated)
    if not gt_norm:
        return 0.0
    return 1.0 if gt_norm in gen_norm else 0.0


def score_mixed(generated: str, ground_truth: str, question: str) -> tuple[float, str]:
    letter_score = score_mcq(generated, ground_truth, question)
    if letter_score is not None:
        return letter_score, "mcq_letter"
    return score_open_ended(generated, ground_truth), "heuristic_contains"


def score_example(benchmark: str, ex: dict) -> tuple[float | None, str]:
    """Return (score, eval_method) for one example, or (None, "unscorable")."""
    generated, ground_truth = ex["generated_answer"], ex["ground_truth"]
    question = ex.get("question", "")

    if is_error(generated):
        return 0.0, "model_error"

    if benchmark in MCQ_BENCHMARKS:
        score = score_mcq(generated, ground_truth, question)
        return (score, "mcq_letter") if score is not None else (None, "unscorable")
    if benchmark in YESNO_BENCHMARKS:
        score = score_yesno(generated, ground_truth)
        return (score, "exact_match") if score is not None else (None, "unscorable")
    if benchmark in SHORT_ANSWER_BENCHMARKS:
        return score_short_answer(generated, ground_truth), "normalized_match"
    if benchmark in OPEN_ENDED_BENCHMARKS:
        return score_open_ended(generated, ground_truth), "heuristic_contains"
    if benchmark in MIXED_BENCHMARKS:
        return score_mixed(generated, ground_truth, question)

    # Unknown benchmark: fall back to the short-answer heuristic.
    return score_short_answer(generated, ground_truth), "normalized_match_fallback"


def iter_result_files(results_dir: Path):
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for benchmark_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for path in sorted(benchmark_dir.glob("*.json")):
                if path.name.endswith("_partial.json"):
                    continue
                yield model_dir.name, benchmark_dir.name, path


def score_results(results_dir: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    for model, benchmark, path in iter_result_files(results_dir):
        with open(path) as f:
            examples = json.load(f)

        scored = []
        for ex in examples:
            score, eval_method = score_example(benchmark, ex)
            ex = {**ex, "score": score, "eval_method": eval_method}
            scored.append(ex)

        out_path = out_dir / model / benchmark / path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(scored, f, indent=2)

        n = len(scored)
        n_error = sum(1 for ex in scored if ex["eval_method"] == "model_error")
        n_unscorable = sum(1 for ex in scored if ex["score"] is None)
        scorable = [ex["score"] for ex in scored if ex["score"] is not None]
        accuracy = sum(scorable) / len(scorable) if scorable else float("nan")

        rows.append({
            "model": model,
            "benchmark": benchmark,
            "n": n,
            "accuracy": accuracy,
            "error_rate": n_error / n if n else float("nan"),
            "unscorable_rate": n_unscorable / n if n else float("nan"),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_scored"))
    args = parser.parse_args()

    df = score_results(args.results_dir, args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.csv"
    df.sort_values(["model", "benchmark"]).to_csv(summary_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\n=== Accuracy by model x benchmark ===")
    print((df.pivot(index="model", columns="benchmark", values="accuracy") * 100).round(1))

    print("\n=== Mean accuracy across benchmarks (models ranked) ===")
    print((df.groupby("model")["accuracy"].mean() * 100).round(1).sort_values(ascending=False))

    print("\n=== Model error rate (raw generation errors, e.g. crashes) ===")
    err = (df.groupby("model")["error_rate"].mean() * 100).round(1).sort_values(ascending=False)
    print(err[err > 0])

    unscorable = df[df["unscorable_rate"] > 0.05]
    if not unscorable.empty:
        print("\n=== Benchmark/model pairs with >5% unscorable (no letter/answer extracted) ===")
        print(unscorable[["model", "benchmark", "unscorable_rate"]].sort_values("unscorable_rate", ascending=False))

    print(f"\nWrote per-example scored results to {args.out_dir}/ and summary to {summary_path}")


if __name__ == "__main__":
    main()
