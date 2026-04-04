#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def _infer_k_from_results_csv(results_csv: Path) -> int:
    m = re.search(r"results_(\d+)\.csv$", results_csv.name)
    if m:
        return int(m.group(1))
    parent = results_csv.parent.name
    m = re.fullmatch(r"k(\d+)", parent)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot infer K from results CSV path: {results_csv}")


def _infer_bsmr_dir_from_results_csv(results_csv: Path) -> Path:
    base = results_csv.parent.parent
    return base / "BSMR_results"


def _infer_out_dir_from_results_csv(results_csv: Path, k: int) -> Path:
    base = results_csv.parent.parent
    return base / "numerical_error_analysis" / f"k{k}"


LOG_NAME_RE = re.compile(r"BSMR_k_(\d+)_a_([\d.]+)_d_([\d.]+)\.log$", re.IGNORECASE)

BLOCK_PATTERNS = {
    "file": r"\[File : ([^\]]+)\]",
    "bsmr_gflops_log": r"\[bsmr_gflops : ([^\]]+)\]",
    "bsmr_alpha": r"\[bsmr_alpha : ([^\]]+)\]",
    "bsmr_delta": r"\[bsmr_delta : ([^\]]+)\]",
    "accuracy_bsmr_vs_cusparse_num_entries_compared": r"\[accuracy_bsmr_vs_cusparse_num_entries_compared : ([^\]]+)\]",
    "accuracy_bsmr_vs_cusparse_frobenius_norm_of_difference": r"\[accuracy_bsmr_vs_cusparse_frobenius_norm_of_difference : ([^\]]+)\]",
    "accuracy_bsmr_vs_cusparse_relative_frobenius_error": r"\[accuracy_bsmr_vs_cusparse_relative_frobenius_error : ([^\]]+)\]",
    "accuracy_bsmr_vs_cusparse_max_relative_diff_per_entry": r"\[accuracy_bsmr_vs_cusparse_max_relative_diff_per_entry : ([^\]]+)\]",
    "accuracy_bsmr_vs_cusparse_root_mean_square_error": r"\[accuracy_bsmr_vs_cusparse_root_mean_square_error : ([^\]]+)\]",
    "accuracy_bsmr_vs_cusparse_num_entries_outside_tolerance": r"\[accuracy_bsmr_vs_cusparse_num_entries_outside_tolerance : ([^\]]+)\]",
    "accuracy_bsmr_vs_cusparse_percent_entries_outside_tolerance": r"\[accuracy_bsmr_vs_cusparse_percent_entries_outside_tolerance : ([^\]]+)%\]",
}


DETAIL_COLUMNS = [
    "file",
    "matrix",
    "M",
    "N",
    "NNZ",
    "Sparsity",
    "K",
    "BSMR",
    "cuSPARSE",
    "bsmr_alpha",
    "bsmr_delta",
    "bsmr_log_basename",
    "bsmr_gflops_log",
    "bsmr_vs_results_csv",
    "accuracy_bsmr_vs_cusparse_num_entries_compared",
    "accuracy_bsmr_vs_cusparse_frobenius_norm_of_difference",
    "accuracy_bsmr_vs_cusparse_relative_frobenius_error",
    "accuracy_bsmr_vs_cusparse_max_relative_diff_per_entry",
    "accuracy_bsmr_vs_cusparse_root_mean_square_error",
    "accuracy_bsmr_vs_cusparse_num_entries_outside_tolerance",
    "accuracy_bsmr_vs_cusparse_percent_entries_outside_tolerance",
]


def _f(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fmt(x: float | int | None) -> str:
    if x is None:
        return ""
    if isinstance(x, int):
        return str(x)
    return f"{x:.10g}"


def _matrix_name(path: str) -> str:
    base = path.rstrip("/").split("/")[-1]
    return base[:-4] if base.endswith(".mtx") else base


def _gflops_match_status(csv_val: float | None, log_val: float | None) -> str:
    if csv_val is None or log_val is None:
        return "missing"
    tol = 0.5 + 1e-4 * max(abs(csv_val), abs(log_val))
    return "ok" if abs(csv_val - log_val) <= tol else "mismatch"


def _percentile(vals: list[float], pct: float) -> float | None:
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    vals = sorted(vals)
    pos = (len(vals) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def parse_log_grid(k: int, bsmr_dir: Path) -> dict[str, list[dict[str, str]]]:
    grid: dict[str, list[dict[str, str]]] = {}
    for log_path in sorted(bsmr_dir.glob(f"BSMR_k_{k}_a_*_d_*.log")):
        m = LOG_NAME_RE.search(log_path.name)
        if not m or int(m.group(1)) != k:
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for block in re.split(r"(?:^|\n)---New data---\n", text):
            if "[File :" not in block:
                continue
            row: dict[str, str] = {"bsmr_log_basename": log_path.name}
            ok = True
            for key, pattern in BLOCK_PATTERNS.items():
                mm = re.search(pattern, block)
                if key == "file":
                    if not mm:
                        ok = False
                        break
                    row[key] = mm.group(1).strip()
                else:
                    row[key] = mm.group(1).strip() if mm else ""
            if not ok:
                continue
            grid.setdefault(row["file"], []).append(row)
    return grid


def select_final_entry(entries: list[dict[str, str]], csv_bsmr: float | None) -> dict[str, str] | None:
    if not entries:
        return None
    if csv_bsmr is None:
        return max(entries, key=lambda e: _f(e.get("bsmr_gflops_log")) or float("-inf"))

    def sort_key(entry: dict[str, str]) -> tuple[int, float, float]:
        log_v = _f(entry.get("bsmr_gflops_log"))
        if log_v is None:
            return (2, float("inf"), float("inf"))
        status = _gflops_match_status(csv_bsmr, log_v)
        penalty = 0 if status == "ok" else 1
        return (penalty, abs(log_v - csv_bsmr), -log_v)

    return min(entries, key=sort_key)


def build_detail_rows(results_csv: Path, bsmr_dir: Path, k: int) -> list[dict[str, str]]:
    grid = parse_log_grid(k, bsmr_dir)
    rows = list(csv.DictReader(results_csv.open(newline="", encoding="utf-8")))
    detail_rows: list[dict[str, str]] = []

    for src in rows:
        out = {key: src.get(key, "") for key in ["file", "M", "N", "NNZ", "Sparsity", "K", "BSMR", "cuSPARSE"]}
        out["matrix"] = _matrix_name(src.get("file", ""))

        bsmr = _f(src.get("BSMR"))
        entry = select_final_entry(grid.get(src.get("file", ""), []), bsmr)
        if entry is None:
            for key in DETAIL_COLUMNS:
                out.setdefault(key, "")
            out["bsmr_vs_results_csv"] = "no_log"
            detail_rows.append(out)
            continue

        for key in DETAIL_COLUMNS:
            if key in entry:
                out[key] = entry[key]

        log_gflops = _f(entry.get("bsmr_gflops_log"))
        out["bsmr_vs_results_csv"] = _gflops_match_status(bsmr, log_gflops)

        for key in DETAIL_COLUMNS:
            out.setdefault(key, "")
        detail_rows.append(out)

    return detail_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_summary_rows(detail_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, float | int | None]]:
    metrics = [
        "accuracy_bsmr_vs_cusparse_relative_frobenius_error",
        "accuracy_bsmr_vs_cusparse_root_mean_square_error",
        "accuracy_bsmr_vs_cusparse_max_relative_diff_per_entry",
        "accuracy_bsmr_vs_cusparse_num_entries_outside_tolerance",
        "accuracy_bsmr_vs_cusparse_percent_entries_outside_tolerance",
    ]
    summary_rows: list[dict[str, str]] = []
    values_map: dict[str, list[float]] = {}

    for metric in metrics:
        vals = [_f(r.get(metric)) for r in detail_rows]
        vals = [v for v in vals if v is not None]
        values_map[metric] = vals
        row = {
            "metric": metric,
            "count": str(len(vals)),
            "median": _fmt(statistics.median(vals) if vals else None),
            "p95": _fmt(_percentile(vals, 0.95)),
            "max": _fmt(max(vals) if vals else None),
        }
        summary_rows.append(row)

    total = len(detail_rows)
    ok_matches = sum(1 for r in detail_rows if r.get("bsmr_vs_results_csv") == "ok")
    mismatch_matches = sum(1 for r in detail_rows if r.get("bsmr_vs_results_csv") == "mismatch")
    missing_logs = sum(1 for r in detail_rows if r.get("bsmr_vs_results_csv") == "no_log")
    zero_outside = sum(
        1
        for r in detail_rows
        if _f(r.get("accuracy_bsmr_vs_cusparse_num_entries_outside_tolerance")) == 0.0
    )
    positive_outside = sum(
        1
        for r in detail_rows
        if (_f(r.get("accuracy_bsmr_vs_cusparse_num_entries_outside_tolerance")) or 0.0) > 0.0
    )

    overview = {
        "metric": "dataset_overview",
        "count": str(total),
        "median": "",
        "p95": "",
        "max": "",
        "matched_logs": str(ok_matches),
        "mismatched_logs": str(mismatch_matches),
        "missing_logs": str(missing_logs),
        "zero_outside_tol_matrices": str(zero_outside),
        "positive_outside_tol_matrices": str(positive_outside),
        "zero_outside_tol_ratio": _fmt((zero_outside / total) if total else None),
    }
    summary_rows.append(overview)

    rel_err = values_map["accuracy_bsmr_vs_cusparse_relative_frobenius_error"]
    rmse = values_map["accuracy_bsmr_vs_cusparse_root_mean_square_error"]

    stats = {
        "num_matrices": total,
        "matched_logs": ok_matches,
        "mismatched_logs": mismatch_matches,
        "missing_logs": missing_logs,
        "zero_outside_tol_matrices": zero_outside,
        "positive_outside_tol_matrices": positive_outside,
        "median_relative_frobenius_error": statistics.median(rel_err) if rel_err else None,
        "p95_relative_frobenius_error": _percentile(rel_err, 0.95),
        "max_relative_frobenius_error": max(rel_err) if rel_err else None,
        "median_rmse": statistics.median(rmse) if rmse else None,
        "p95_rmse": _percentile(rmse, 0.95),
    }
    return summary_rows, stats


def _plot_jitter_scatter_with_markers(
    values: list[float],
    title: str,
    out_base: Path,
    median_v: float | None,
    p95_v: float | None,
    max_v: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    rng = random.Random(0)
    ys = list(values)
    xs = [1.0 + rng.uniform(-0.34, 0.34) for _ in ys]
    ax.scatter(xs, ys, s=14, alpha=0.7, edgecolors="none", color="#4C78A8")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlim(0.60, 1.40)
    ax.set_xticks([])
    marker_specs = [
        (median_v, "Median", "#F58518"),
        (p95_v, "95th percentile", "#54A24B"),
        (max_v, "Maximum", "#E45756"),
    ]
    added = False
    for value, label, color in marker_specs:
        if value is None or value <= 0:
            continue
        ax.axhline(value, color=color, linestyle="--", linewidth=1.5, label=f"{label}: {value:.2e}")
        added = True
    if added:
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.58, 0.18))
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_main_figure(detail_rows: list[dict[str, str]], out_dir: Path, stats: dict[str, float | int | None], k: int) -> Path | None:
    rel_err = [
        _f(r.get("accuracy_bsmr_vs_cusparse_relative_frobenius_error"))
        for r in detail_rows
    ]
    rel_err = [v for v in rel_err if v is not None and v > 0]
    if not rel_err:
        return None
    figure_base = out_dir / f"k{k}_relative_frobenius_error_distribution"
    _plot_jitter_scatter_with_markers(
        rel_err,
        f"Relative Frobenius Error on {len(rel_err)} K={k} Matrices",
        figure_base,
        stats.get("median_relative_frobenius_error"),
        stats.get("p95_relative_frobenius_error"),
        stats.get("max_relative_frobenius_error"),
    )
    return figure_base.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize BSMR numerical error against cuSPARSE. "
        "If only --results-csv is provided, --bsmr-dir, --out-dir, and --k are inferred automatically."
    )
    parser.add_argument("--results-csv", type=Path, required=True, help="Path to results_<K>.csv")
    parser.add_argument("--bsmr-dir", type=Path, default=None, help="Path to BSMR_results (optional)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (optional)")
    parser.add_argument("--k", type=int, default=None, help="K value encoded in log filenames (optional)")
    args = parser.parse_args()

    results_csv = args.results_csv.resolve()
    k = args.k if args.k is not None else _infer_k_from_results_csv(results_csv)
    bsmr_dir = args.bsmr_dir.resolve() if args.bsmr_dir is not None else _infer_bsmr_dir_from_results_csv(results_csv)
    out_dir = args.out_dir.resolve() if args.out_dir is not None else _infer_out_dir_from_results_csv(results_csv, k)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = build_detail_rows(results_csv, bsmr_dir, k)

    summary_rows, stats = build_summary_rows(detail_rows)
    summary_csv = out_dir / f"k{k}_numerical_error_summary.csv"
    write_csv(
        summary_csv,
        summary_rows,
        [
            "metric",
            "count",
            "median",
            "p95",
            "max",
            "matched_logs",
            "mismatched_logs",
            "missing_logs",
            "zero_outside_tol_matrices",
            "positive_outside_tol_matrices",
            "zero_outside_tol_ratio",
        ],
    )

    figure_pdf = write_main_figure(detail_rows, out_dir, stats, k)

    print(f"Saved summary CSV: {summary_csv}")
    if figure_pdf is not None:
        print(f"Saved main figure: {figure_pdf}")


if __name__ == "__main__":
    main()
