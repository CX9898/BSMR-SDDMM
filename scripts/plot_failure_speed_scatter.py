#!/usr/bin/env python3
"""K=128 failure-analysis scatter plot for the BSMR paper (Figure 8).

Generates a 2x3 per-matrix scatter relating BSMR's speedup to two
reordering-related indicators (Tensor-Core data share, block-density gain)
across three baselines (RoDe, ASpT, FlashSparse).

Outputs (in --out-dir, default deep_failure_analysis/):
  k128_failure_metric_scatter.png
  k128_failure_metric_scatter.pdf

Caption-ready statistics (Spearman rho, p-value, win count, clip count,
max speedup) are printed to stdout.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLOR_FAST = "#4C78A8"
COLOR_SLOW = "#F58518"
PANEL_ORDER = ["RoDe", "ASpT", "FlashSparse"]
PANEL_TAG = {"RoDe": "CUDA-Core", "ASpT": "CUDA-Core", "FlashSparse": "TC-based"}
CLIP_Y = 4.0


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_case_csv() -> Path:
    return _scripts_dir() / "results_suiteSparse_dataset" / "deep_failure_analysis" / "per_case_diagnostics.csv"


def _default_out_dir() -> Path:
    return _scripts_dir() / "results_suiteSparse_dataset" / "deep_failure_analysis"


def _to_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _load_per_baseline(
    case_csv: Path, baselines: list[str]
) -> dict[str, list[dict[str, float]]]:
    """Return {baseline: [ {speedup, dense_data_share, density_gain}, ... ]}.

    Filters to K=128 rows; drops rows with non-positive speedup or missing
    dense_data_share. Missing density_gain is treated as 0.0.
    """
    rows = list(csv.DictReader(case_csv.open(newline="", encoding="utf-8")))
    out: dict[str, list[dict[str, float]]] = {b: [] for b in baselines}
    for r in rows:
        if r.get("K") != "128":
            continue
        b = r.get("baseline")
        if b not in out:
            continue
        sp = _to_float(r.get("speedup_over_baseline"))
        dds = _to_float(r.get("dense_data_share"))
        dg = _to_float(r.get("density_gain"))
        if sp is None or sp <= 0 or dds is None:
            continue
        out[b].append({
            "speedup": sp,
            "dense_data_share": dds,
            "density_gain": dg if dg is not None else 0.0,
        })
    return out


def _binned_geomean(
    x: np.ndarray, y: np.ndarray, n_bins: int, x_min: float, x_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Geometric-mean of y per equal-width X bin; bins with <3 points dropped."""
    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers, gm = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (x >= lo) & (x <= hi) if i == n_bins - 1 else (x >= lo) & (x < hi)
        if mask.sum() < 3:
            continue
        centers.append(0.5 * (lo + hi))
        gm.append(float(np.exp(np.log(np.clip(y[mask], 1e-9, None)).mean())))
    return np.asarray(centers), np.asarray(gm)


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy import stats
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def plot_k128_failure_scatter(case_csv: Path, out_dir: Path) -> tuple[Path, Path, list[str]]:
    """Build the 2x3 failure-analysis scatter and save PNG + PDF.

    Returns (png_path, pdf_path, caption_facts).
    """
    data = _load_per_baseline(case_csv, baselines=PANEL_ORDER)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), sharey=True)
    plt.rcParams.update({"font.size": 10})

    metric_specs = [
        ("dense_data_share", "Data share routed to Tensor Cores",
         -0.02, 1.02, 0.0, 1.0, 8, None),
        ("density_gain", "Block-density gain after reordering",
         -0.4, 1.0, -0.4, 1.0, 7, 0.0),
    ]

    caption_facts: list[str] = []

    for col, baseline in enumerate(PANEL_ORDER):
        recs = data[baseline]
        y_arr = np.asarray([r["speedup"] for r in recs])
        n_total = len(y_arr)
        n_fast = int((y_arr >= 1.0).sum())
        y_max_full = float(y_arr.max())
        n_clipped = int((y_arr > CLIP_Y).sum())

        for row_idx, (key, xlabel, x_lo, x_hi, bin_lo, bin_hi, n_bins, vline) in enumerate(metric_specs):
            ax = axes[row_idx][col]
            x = np.asarray([r[key] for r in recs])
            colors = np.where(y_arr >= 1.0, COLOR_FAST, COLOR_SLOW)

            in_mask = y_arr <= CLIP_Y
            over_mask = ~in_mask

            ax.scatter(x[in_mask], y_arr[in_mask], s=14, c=colors[in_mask],
                       alpha=0.55, edgecolors="none", rasterized=True)

            if over_mask.sum() > 0:
                ax.scatter(
                    x[over_mask],
                    np.full(over_mask.sum(), CLIP_Y * 0.97),
                    s=42, c=colors[over_mask], marker="^",
                    edgecolors="white", linewidths=0.6,
                    alpha=0.95, clip_on=False, zorder=5,
                )

            bx, by = _binned_geomean(x, y_arr, n_bins=n_bins, x_min=bin_lo, x_max=bin_hi)
            if len(bx) >= 2:
                ax.plot(bx, np.minimum(by, CLIP_Y), color="black",
                        linewidth=1.4, marker="o", markersize=4)

            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
            if vline is not None:
                ax.axvline(vline, color="0.5", linestyle=":", linewidth=0.8, alpha=0.7)

            rho, p = _spearman(x, y_arr)
            ax.text(
                0.96, 0.96, f"$\\rho$ = {rho:+.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.7", alpha=0.85),
            )

            ax.set_ylim(0.0, CLIP_Y)
            ax.set_xlim(x_lo, x_hi)
            ax.grid(True, linestyle="--", alpha=0.25)
            if row_idx == 0:
                ax.set_title(f"{baseline}  ({PANEL_TAG[baseline]})")
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel("BSMR / baseline speedup")

            metric_short = "data share" if key == "dense_data_share" else "density gain"
            caption_facts.append(
                f"  {baseline} vs {metric_short}: rho={rho:+.2f}, p={p:.1e}"
            )

        caption_facts.append(
            f"  -> {baseline}: BSMR faster on {n_fast}/{n_total} matrices; "
            f"{n_clipped} pts above {CLIP_Y:.0f}x (max {y_max_full:.1f}x)"
        )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLOR_FAST,
               markeredgecolor="none", markersize=7, label="BSMR $\\geq$ baseline"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLOR_SLOW,
               markeredgecolor="none", markersize=7, label="BSMR < baseline"),
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="gray",
               markeredgecolor="white", markersize=8,
               label=f"Clipped (speedup $>$ {CLIP_Y:.0f}x)"),
        Line2D([0], [0], color="black", linewidth=1.4, marker="o", markersize=4,
               label="Geometric-mean trend"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0,
               label="Speedup = 1.0"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "k128_failure_metric_scatter.png"
    pdf_path = out_dir / "k128_failure_metric_scatter.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, caption_facts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K=128 failure-analysis scatter plot for BSMR paper Figure 8.",
    )
    parser.add_argument("--case-csv", type=Path, default=_default_case_csv(),
                        help="Path to per_case_diagnostics.csv")
    parser.add_argument("--out-dir", type=Path, default=_default_out_dir(),
                        help="Output directory")
    args = parser.parse_args()

    png_path, pdf_path, facts = plot_k128_failure_scatter(
        args.case_csv.resolve(), args.out_dir.resolve()
    )
    print(f"Saved PNG to {png_path}")
    print(f"Saved PDF to {pdf_path}")
    print()
    print("Caption-ready statistics:")
    for line in facts:
        print(line)


if __name__ == "__main__":
    main()
