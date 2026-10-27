# Experiment and analysis entry points

These scripts are independent of the GitHub Pages site in `docs/`. Shared Python implementation stays in `src/`; notebooks are in `notebooks/`, with the instance-level attention notebook alongside its companion scripts in `scripts/attention_analysis/`.

Run from the repository root after installing the project and analysis requirements. The entry points below also resolve the repository root when called by absolute path from another directory.

| Entry point | Purpose |
| --- | --- |
| `main_field.py` | Canvas-padding experiment and matched resized/bicubic controls |
| `main_interpolation.py` | Bicubic pixel-resolution scaling used in the paper |
| `main_scale.py` | Original Real-ESRGAN runner referenced by the restored README; a legacy alternative, not the paper's final bicubic protocol |
| `main_attention.py` | Attention-map analysis |
| `main_cka.py` | Layerwise representation/CKA analysis |
| `trajectory.py` | Convert evaluated result files to correctness trajectories |
| `attention_analysis/compute_attention.py` | Per-instance attention extraction used by the interactive notebook |
| `attention_analysis/run_mmvet_attention_analysis.py` | MMVet attention diagnostics |

Examples:

```sh
python scripts/main_interpolation.py --model_name Qwen/Qwen2-VL-2B-Instruct --benchmark MMVet --added_scales '1,1.5,2,3,4,5,6' --num_gpus 1 --output_path results_interpolation_new/Qwen_Qwen2-VL-2B-Instruct/MMVet
python scripts/main_field.py --model_name Qwen/Qwen2-VL-2B-Instruct --benchmark MMVet --added_fields '0,0.5,1,2,3,4,5' --position center --num_gpus 1 --output_path results/Qwen_Qwen2-VL-2B-Instruct/MMVet
python scripts/main_attention.py --help
python scripts/main_cka.py --help
python scripts/trajectory.py --help
```

The drivers also support module invocation, for example `python -m scripts.main_field --help`. The attention companion scripts are intended for direct invocation; their local `common.py` handles model configuration.

Evaluation implementation remains in `src/`:

```sh
python -m src.parse_results --results-dir results --out-dir results_scored
python -m src.parse_interpolation_results --results-dir results_interpolation_new --out-dir results_scored_interpolation
```

These parsers use diagnostic heuristics. The legacy LLM judges (`python src/evaluate.py`, etc.) retain their original configurations and require reconciliation with the paper before paper-faithful evaluation. Consult `research-docs/NOTEBOOK_AUDIT.md` for missing inputs, metric-definition differences, and known plotting/evaluation issues. No new results were generated during organization.

The six historical shell sweeps, two legacy API inference drivers, and unused S3 cache helper were removed from this repository and preserved in the sibling `Waldo/legacy_scripts/` directory. No SLURM job files or cluster launchers are part of this scripts directory.
