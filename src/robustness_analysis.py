"""Resolution/compression robustness analysis over the field/resized/bicubic
scale sweep in results_scored_field/.

Data source and scope (verified against the files on disk, not assumed):

  * The "scale" axis is the ``field`` variant of the field-experiment triplet
    (src/utils.py:build_field_images / main_field.py): the original image
    embedded in a progressively larger white-bordered canvas, at field
    percent values 0, 0.5, 1, 2, 3, 4, 5. The white border is the
    "distractor" described in the task.
  * The two controls are ``resized`` (the real content alone, downscaled by
    the exact coefficient the model's own preprocessor applied to the padded
    canvas - the border stripped back out) and ``bicubic`` (``resized``
    upsampled back to the original resolution via bicubic interpolation, so
    detail is lost without a canvas-size change). Both strip the white-field
    padding; a "no_recovery" case only becomes "distractor-induced" if EITHER
    control gets it right, matching the field/resized/bicubic docstring in
    src/utils.py and the "Add resized and bicubic control experiments to
    field run" commit.
  * Model scope is deliberately restricted to ``llava-hf_llava-1.5-7b-hf`` and
    ``Qwen_Qwen2-VL-7B-Instruct``. ``results_scored/`` (unsuffixed) predates
    the "CKA and bug fix for field" commit (8fe1af9) and was confirmed broken
    for this axis (constant token count/score across field=0.5..5 for
    spot-checked benchmarks - the transformation silently wasn't applied).
    ``meta-llama_Llama-3.2-11B-Vision-Instruct`` does have post-fix data, but
    only two scale points (field=0, 0.5) exist for its one benchmark (MMVet)
    and neither carries vision-token metadata - too sparse for a scale
    sequence, so it is excluded rather than silently padded with NaNs.
  * field=0 is pulled from a separate legacy run (results_old/) that predates
    the field/resized/bicubic split and used different preprocessing
    defaults: its token counts are 1-2 orders of magnitude smaller than
    field=0.5's, an artifact of that regime rather than genuine compression.
    Per explicit user direction it is KEPT as the s_1 anchor of the ordinal
    scale sequence (used for the preceding/recovered walk), but EXCLUDED from
    the compression-ratio metric (which is anchored at field=0.5 instead) -
    see `compute_compression_ratio`.

Usage:
    python -m src.robustness_analysis
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

RESULTS_DIR = Path("results_scored_field")
OUT_DIR = Path("results_robustness")

SCALES = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
VARIANTS = ["field", "resized", "bicubic"]
COMPRESSION_REF_SCALE = 0.5  # anchor for the compression-ratio metric; see module docstring

MODELS = ["llava-hf_llava-1.5-7b-hf", "Qwen_Qwen2-VL-7B-Instruct"]

# dataviz palette (references/palette.md): fixed categorical slot order.
COLOR_MODEL_A = "#2a78d6"   # slot 1 blue
COLOR_MODEL_B = "#eb6834"   # slot 2 orange
COLOR_TRUE_LOSS = "#e34948"      # slot 8 red - "still wrong under every control"
COLOR_DISTRACTOR = "#1baf7a"     # slot 3 aqua - "the border, not compression, was the cause"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def _scale_tag(scale: float) -> str:
    return str(int(scale)) if float(scale).is_integer() else str(scale)


def _model_short(model: str) -> str:
    return model.split("_", 1)[1] if "_" in model else model


def _variant_path(model: str, variant: str, benchmark: str, scale: float) -> Path:
    tag = _scale_tag(scale)
    short = _model_short(model)
    return RESULTS_DIR / model / variant / benchmark / f"{benchmark}_{short}_{variant}{tag}.json"


def _benchmarks_for(model: str) -> list[str]:
    field_dir = RESULTS_DIR / model / "field"
    return sorted(p.name for p in field_dir.iterdir() if p.is_dir())


def _load_examples_by_key(path: Path) -> dict[tuple, dict]:
    """Load a result file keyed by (id, question), dropping exact duplicate
    rows (a handful of benchmarks, e.g. MME-RealWorld-Lite, contain literal
    duplicate rows on disk - same id, question, ground truth AND answer)."""
    if not path.exists():
        return {}
    with open(path) as f:
        examples = json.load(f)
    by_key = {}
    for ex in examples:
        key = (ex["id"], ex["question"])
        by_key.setdefault(key, ex)
    return by_key


def load_model_benchmark(model: str, benchmark: str) -> list[dict]:
    """One row per (question, scale) for the field axis, enriched with the
    resized/bicubic control correctness and the resized-control token count
    (the effective-resolution-after-patchification signal, isolated from the
    padding distractor - see module docstring)."""
    per_scale = {
        scale: {variant: _load_examples_by_key(_variant_path(model, variant, benchmark, scale))
                for variant in VARIANTS}
        for scale in SCALES
    }

    all_keys = set()
    for scale in SCALES:
        all_keys |= set(per_scale[scale]["field"].keys())

    rows = []
    for key in all_keys:
        for scale in SCALES:
            field_ex = per_scale[scale]["field"].get(key)
            if field_ex is None:
                continue
            score = field_ex.get("score")
            if score is None:
                continue  # unscorable (~10 rows total across the whole dataset)

            resized_ex = per_scale[scale]["resized"].get(key)
            bicubic_ex = per_scale[scale]["bicubic"].get(key)

            def _ctrl_correct(ex):
                if ex is None or ex.get("score") is None:
                    return None
                return bool(round(ex["score"]))

            tokens_resized = field_ex.get("num_vision_tokens_resized", field_ex.get("num_vision_tokens"))

            rows.append({
                "model": model,
                "benchmark": benchmark,
                "question_id": key[0],
                "question": key[1],
                "scale": scale,
                "correct": bool(round(score)),
                "resized_correct": _ctrl_correct(resized_ex),
                "bicubic_correct": _ctrl_correct(bicubic_ex),
                "tokens_resized": tokens_resized,
            })
    return rows


def load_all() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for benchmark in _benchmarks_for(model):
            rows.extend(load_model_benchmark(model, benchmark))
    df = pd.DataFrame(rows)
    df["question_key"] = df["model"] + "|" + df["benchmark"] + "|" + df["question_id"].astype(str) + "|" + df["question"]
    return df


def compute_compression_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """compression_ratio = tokens_resized(scale=0.5) / tokens_resized(scale),
    computed per (model, benchmark, question) so it is normalized against
    that exact image's own token count at the anchor scale - not a
    population average, and not comparable in absolute terms across models
    (different architectures patchify at very different token densities;
    only the within-model ordering/ratio is meaningful, per the task's own
    "not raw pixel resolution" instruction).

    field=0 rows get compression_ratio = NaN: they come from a different
    preprocessing regime (see module docstring) where token counts are not
    comparable to the 0.5-5 sweep, so folding them into the ratio would
    silently misrepresent them as "more compressed" purely as an artifact of
    a different pipeline default, not genuine compression.
    """
    ref = (
        df[df["scale"] == COMPRESSION_REF_SCALE]
        .set_index("question_key")["tokens_resized"]
    )
    ref_tokens = df["question_key"].map(ref)
    ratio = ref_tokens / df["tokens_resized"]
    ratio = ratio.where(df["scale"] != 0.0, np.nan)
    df = df.copy()
    df["compression_ratio"] = ratio
    return df


def bin_compression_ratio(df: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    """Per-model quantile (tertile) binning of compression_ratio, stated
    explicitly rather than hardcoding fixed cutoffs, since the achievable
    range differs enormously by architecture (dynamic-resolution models like
    Qwen2-VL span a huge token range; fixed-budget encoders may barely move).

    llava-1.5-7b-hf is a documented degenerate case: it always resizes/crops
    to a fixed 336x336 CLIP input, so num_vision_tokens_resized is exactly
    576 at every scale and compression_ratio is exactly 1.0 everywhere -
    qcut cannot form 3 distinct bins from a constant column. That model is
    reported as a single "no token-count variation" bin with an explicit
    warning, rather than failing or silently collapsing without comment.
    """
    df = df.copy()
    df["compression_bin"] = pd.Series(pd.NA, index=df.index, dtype="object")
    for model, sub in df.groupby("model"):
        valid = sub["compression_ratio"].dropna()
        if valid.nunique() <= 1:
            warnings.warn(
                f"{model}: compression_ratio has no variation (token count constant "
                f"across scales) - reporting a single degenerate bin instead of tertiles."
            )
            df.loc[sub.index[sub["compression_ratio"].notna()], "compression_bin"] = "All (no token-count variation)"
            continue
        try:
            labels = ["Low compression", "Mid compression", "High compression"][:n_bins]
            binned = pd.qcut(valid, q=n_bins, labels=labels, duplicates="drop")
        except ValueError:
            binned = pd.qcut(valid, q=valid.nunique(), labels=False, duplicates="drop")
        df.loc[binned.index, "compression_bin"] = binned.astype(str)
    return df


def classify_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, benchmark, question): walk the ordered scale sequence and
    assign, for every incorrect scale point, preceding_correct/recovered/
    event_class per the task's step-3 definitions. Also records, per
    question, the last-ever-correct scale, the drop point immediately after
    it, and a non-monotonicity flag (>=2 correctness flips - i.e. strictly
    more than the single "degrades and stays degraded" transition)."""
    df = df.sort_values(["model", "benchmark", "question_id", "question", "scale"]).copy()
    for col in ["preceding_correct", "recovered"]:
        df[col] = False
    df["event_class"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df["last_correct_scale"] = np.nan
    df["drop_point_scale"] = np.nan
    df["is_drop_point"] = False
    df["n_flips"] = 0
    df["non_monotonic"] = False

    for key, g in df.groupby("question_key", sort=False):
        idx = g.index.tolist()
        correct = g["correct"].tolist()
        scales = g["scale"].tolist()
        n = len(idx)

        for i in range(n):
            if correct[i]:
                continue
            preceding_correct = any(correct[:i])
            df.at[idx[i], "preceding_correct"] = preceding_correct
            if preceding_correct:
                recovered = any(correct[i + 1:])
                df.at[idx[i], "recovered"] = recovered
                df.at[idx[i], "event_class"] = "recovered" if recovered else "no_recovery"

        correct_scales = [scales[i] for i in range(n) if correct[i]]
        last_correct = max(correct_scales) if correct_scales else None
        later = [s for s in scales if last_correct is not None and s > last_correct]
        drop_point = min(later) if later else None

        flips = sum(1 for i in range(1, n) if correct[i] != correct[i - 1])

        for i in range(n):
            df.at[idx[i], "last_correct_scale"] = last_correct if last_correct is not None else np.nan
            df.at[idx[i], "drop_point_scale"] = drop_point if drop_point is not None else np.nan
            df.at[idx[i], "is_drop_point"] = (drop_point is not None and scales[i] == drop_point)
            df.at[idx[i], "n_flips"] = flips
            df.at[idx[i], "non_monotonic"] = flips >= 2

    return df


def apply_distractor_check(df: pd.DataFrame) -> pd.DataFrame:
    """For no_recovery rows: answerable_without_distractors is True if either
    control (resized or bicubic - both distractor/padding-free) is correct at
    that same scale. Splits genuine loss into true_information_loss (still
    wrong under both controls) vs distractor_induced_failure (an artifact of
    the added border, not of compression)."""
    df = df.copy()
    is_no_recovery = df["event_class"] == "no_recovery"

    any_control_correct = (
        df["resized_correct"].fillna(False) | df["bicubic_correct"].fillna(False)
    )
    both_controls_missing = df["resized_correct"].isna() & df["bicubic_correct"].isna()

    df["answerable_without_distractors"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[is_no_recovery & ~both_controls_missing, "answerable_without_distractors"] = (
        any_control_correct[is_no_recovery & ~both_controls_missing]
    )

    df["true_information_loss"] = is_no_recovery & (df["answerable_without_distractors"] == False)  # noqa: E712
    df["distractor_induced_failure"] = is_no_recovery & (df["answerable_without_distractors"] == True)  # noqa: E712
    return df


def clopper_pearson(successes: int, n: int):
    if n == 0:
        return (np.nan, np.nan)
    return proportion_confint(successes, n, method="beta")


def stratified_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (model, compression_bin) and report N, genuine-loss
    (no_recovery) rate, recovered (noisy-baseline) rate, and - conditional on
    no_recovery - the true-loss vs distractor-induced split, each with a 95%
    Clopper-Pearson CI."""
    rows = []
    scored = df[df["compression_bin"].notna()]
    for (model, cbin), g in scored.groupby(["model", "compression_bin"]):
        n = len(g)
        n_no_recovery = int((g["event_class"] == "no_recovery").sum())
        n_recovered = int((g["event_class"] == "recovered").sum())
        n_true_loss = int(g["true_information_loss"].sum())
        n_distractor = int(g["distractor_induced_failure"].sum())
        n_controls_available = n_true_loss + n_distractor

        no_recovery_ci = clopper_pearson(n_no_recovery, n)
        recovered_ci = clopper_pearson(n_recovered, n)
        true_loss_ci = clopper_pearson(n_true_loss, n_controls_available)
        distractor_ci = clopper_pearson(n_distractor, n_controls_available)

        rows.append({
            "model": model,
            "compression_bin": cbin,
            "n_evaluated": n,
            "no_recovery_rate": n_no_recovery / n if n else np.nan,
            "no_recovery_ci_lo": no_recovery_ci[0], "no_recovery_ci_hi": no_recovery_ci[1],
            "recovered_rate": n_recovered / n if n else np.nan,
            "recovered_ci_lo": recovered_ci[0], "recovered_ci_hi": recovered_ci[1],
            "n_no_recovery": n_no_recovery,
            "n_controls_available": n_controls_available,
            "true_loss_rate_of_no_recovery": n_true_loss / n_controls_available if n_controls_available else np.nan,
            "true_loss_ci_lo": true_loss_ci[0], "true_loss_ci_hi": true_loss_ci[1],
            "distractor_induced_rate_of_no_recovery": n_distractor / n_controls_available if n_controls_available else np.nan,
            "distractor_ci_lo": distractor_ci[0], "distractor_ci_hi": distractor_ci[1],
        })
    return pd.DataFrame(rows).sort_values(["model", "compression_bin"])


def non_monotonic_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per question with a surprising (>=2 flip) correctness
    pattern, with the full ordered correct/incorrect sequence spelled out."""
    rows = []
    flagged = df[df["non_monotonic"]]
    for key, g in flagged.groupby("question_key", sort=False):
        g = g.sort_values("scale")
        pattern = " -> ".join("correct" if c else "incorrect" for c in g["correct"])
        rows.append({
            "model": g["model"].iloc[0],
            "benchmark": g["benchmark"].iloc[0],
            "question_id": g["question_id"].iloc[0],
            "question": g["question"].iloc[0],
            "n_flips": int(g["n_flips"].iloc[0]),
            "pattern": pattern,
            "scales": ", ".join(str(s) for s in g["scale"]),
        })
    return pd.DataFrame(rows).sort_values(["n_flips", "model", "benchmark"], ascending=[False, True, True])


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_genuine_loss_vs_compression(stats: pd.DataFrame, out_path: Path):
    """One line per model: genuine-loss (no_recovery) rate vs compression
    bin, with Clopper-Pearson CI bands. Bins are ordinal per model (Low/Mid/
    High, or the degenerate single bin for llava), so the x-axis is a shared
    category axis rather than a numeric compression-ratio scale."""
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    colors = {MODELS[0]: COLOR_MODEL_A, MODELS[1]: COLOR_MODEL_B}
    order = ["Low compression", "Mid compression", "High compression", "All (no token-count variation)"]

    for model in MODELS:
        sub = stats[stats["model"] == model].copy()
        sub["compression_bin"] = pd.Categorical(sub["compression_bin"], categories=order, ordered=True)
        sub = sub.sort_values("compression_bin")
        x = np.arange(len(sub))
        y = sub["no_recovery_rate"].to_numpy()
        lo = sub["no_recovery_ci_lo"].to_numpy()
        hi = sub["no_recovery_ci_hi"].to_numpy()
        color = colors[model]
        ax.plot(x, y, marker="o", markersize=6, linewidth=2, color=color, label=_model_short(model), zorder=3)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["compression_bin"].astype(str), fontsize=9, color=INK_SECONDARY)

    ax.set_ylabel("Genuine-loss rate (no_recovery)", color=INK_PRIMARY, fontsize=10)
    ax.set_title("Genuine information loss vs. compression level", color=INK_PRIMARY, fontsize=12, pad=12)
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK_PRIMARY, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_true_vs_distractor_stacked(stats: pd.DataFrame, out_path: Path):
    """Stacked bar per (model, compression bin): true-information-loss vs
    distractor-induced share of the no_recovery cases in that group."""
    order = ["Low compression", "Mid compression", "High compression", "All (no token-count variation)"]
    stats = stats.copy()
    stats["compression_bin"] = pd.Categorical(stats["compression_bin"], categories=order, ordered=True)
    stats = stats.sort_values(["model", "compression_bin"])

    labels = [f"{_model_short(m)}\n{b}" for m, b in zip(stats["model"], stats["compression_bin"])]
    x = np.arange(len(stats))

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    true_loss = stats["true_loss_rate_of_no_recovery"].to_numpy()
    distractor = stats["distractor_induced_rate_of_no_recovery"].to_numpy()

    ax.bar(x, true_loss, width=0.6, color=COLOR_TRUE_LOSS, label="True information loss", zorder=3)
    ax.bar(x, distractor, width=0.6, bottom=true_loss, color=COLOR_DISTRACTOR, label="Distractor-induced failure", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel("Share of no_recovery cases", color=INK_PRIMARY, fontsize=10)
    ax.set_title("True information loss vs. distractor-induced failure", color=INK_PRIMARY, fontsize=12, pad=12)
    ax.set_ylim(0, 1.05)
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK_PRIMARY, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("Loading field/resized/bicubic triplet data...")
    df = load_all()
    print(f"Loaded {len(df)} (model, benchmark, question, scale) rows "
          f"across {df['model'].nunique()} models, {df['benchmark'].nunique()} benchmarks, "
          f"{df['question_key'].nunique()} unique questions.")

    df = compute_compression_ratio(df)
    df = bin_compression_ratio(df)
    df = classify_questions(df)
    df = apply_distractor_check(df)

    tidy_path = OUT_DIR / "tidy_long.csv"
    df.drop(columns=["question_key"]).to_csv(tidy_path, index=False)
    print(f"Wrote tidy long-format data to {tidy_path} ({len(df)} rows)")

    stats = stratified_stats(df)
    stats_path = OUT_DIR / "summary_stats.csv"
    stats.to_csv(stats_path, index=False)
    print(f"Wrote stratified summary to {stats_path}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(stats.to_string(index=False))

    nm = non_monotonic_report(df)
    nm_path = OUT_DIR / "non_monotonic_questions.csv"
    nm.to_csv(nm_path, index=False)
    print(f"\n{len(nm)} questions with a non-monotonic (>=2 flip) correctness pattern -> {nm_path}")
    if not nm.empty:
        print(nm.head(10).to_string(index=False))

    plot_genuine_loss_vs_compression(stats, plots_dir / "genuine_loss_vs_compression.png")
    plot_true_vs_distractor_stacked(stats, plots_dir / "true_vs_distractor_stacked.png")
    print(f"\nWrote plots to {plots_dir}/")


if __name__ == "__main__":
    main()
