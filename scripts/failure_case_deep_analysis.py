#!/usr/bin/env python3
"""
更深入的失败/成功案例对照分析。

默认会读取:
  scripts/results_suiteSparse_dataset/k*/results_<K>.csv
并结合:
  scripts/results_suiteSparse_dataset/BSMR_results/BSMR_k_<K>_a_*_d_*.log

输出:
  scripts/results_suiteSparse_dataset/deep_failure_analysis/

关注点:
  1. 失败严重程度 (failure margin)
  2. dense path 是否真正转化为性能
  3. 参数鲁棒性 (最优点附近是否稳定)
  4. 跨 K 一致性
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_results_csv(k: int) -> Path:
    return _scripts_dir() / "results_suiteSparse_dataset" / f"k{k}" / f"results_{k}.csv"


def _default_bsmr_dir() -> Path:
    return _scripts_dir() / "results_suiteSparse_dataset" / "BSMR_results"


def _default_out_dir() -> Path:
    return _scripts_dir() / "results_suiteSparse_dataset" / "deep_failure_analysis"


def _discover_default_csvs() -> list[Path]:
    base = _scripts_dir() / "results_suiteSparse_dataset"
    return sorted(base.glob("k*/results_[0-9]*.csv"))


def _f(x: str | None) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _median(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.6g}"


def _speedup_bin(speedup: float | None) -> str:
    if speedup is None:
        return "unknown"
    if speedup < 1.0:
        return "<1.0x"
    if speedup < 1.5:
        return "1.0~1.5x"
    if speedup < 2.0:
        return "1.5~2.0x"
    return ">=2.0x"


def _matrix_name(file_path: str) -> str:
    base = file_path.rstrip("/").split("/")[-1]
    return base[:-4] if base.endswith(".mtx") else base


def _family_name(file_path: str) -> str:
    m = re.search(r"suiteSparse_dataset/([^/]+)/", file_path)
    return m.group(1) if m else "unknown"


LOG_NAME_RE = re.compile(r"BSMR_k_(\d+)_a_([\d.]+)_d_([\d.]+)\.log$", re.IGNORECASE)

BLOCK_PATTERNS = {
    "file": r"\[File : ([^\]]+)\]",
    "gflops": r"\[bsmr_gflops : ([^\]]+)\]",
    "sddmm_ms": r"\[bsmr_sddmm : ([^\]]+)\]",
    "num_clusters": r"\[bsmr_numClusters : ([^\]]+)\]",
    "dense_block": r"\[bsmr_numDenseBlock : ([^\]]+)\]",
    "dense_density": r"\[bsmr_averageDensity : ([^\]]+)\]",
    "orig_dense_block": r"\[original_numDenseBlock : ([^\]]+)\]",
    "orig_dense_density": r"\[original_averageDensity : ([^\]]+)\]",
    "dense_tb": r"\[bsmr_numDenseThreadBlocks : ([^\]]+)\]",
    "sparse_tb": r"\[bsmr_numSparseThreadBlocks : ([^\]]+)\]",
    "tb_ratio": r"\[bsmr_threadBlockRatio : ([^\]]+)\]",
    "dense_data": r"\[bsmr_numDenseData : ([^\]]+)\]",
    "sparse_data": r"\[bsmr_numSparseData : ([^\]]+)\]",
    "data_ratio": r"\[bsmr_dataRatio: ([^\]]+)\]",
}


def load_bsmr_grid_for_k(k: int, bsmr_dir: Path) -> dict[str, list[dict[str, float | str | None]]]:
    grid: dict[str, list[dict[str, float | str | None]]] = defaultdict(list)
    if not bsmr_dir.is_dir():
        return grid

    for log_path in sorted(bsmr_dir.glob(f"BSMR_k_{k}_a_*_d_*.log")):
        m = LOG_NAME_RE.search(log_path.name)
        if not m:
            continue
        k_file, alpha_s, delta_s = m.groups()
        if int(k_file) != k:
            continue
        alpha = float(alpha_s)
        delta = float(delta_s)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for block in re.split(r"(?:^|\n)---New data---\n", text):
            if "[File :" not in block:
                continue
            file_m = re.search(BLOCK_PATTERNS["file"], block)
            g_m = re.search(BLOCK_PATTERNS["gflops"], block)
            if not file_m or not g_m:
                continue
            row: dict[str, float | str | None] = {
                "file": file_m.group(1).strip(),
                "alpha": alpha,
                "delta": delta,
                "log": log_path.name,
                "gflops": _f(g_m.group(1).strip()),
            }
            for key, pattern in BLOCK_PATTERNS.items():
                if key in {"file", "gflops"}:
                    continue
                mm = re.search(pattern, block)
                row[key] = _f(mm.group(1).strip()) if mm else None
            grid[str(row["file"])].append(row)
    return grid


def summarize_grid(points: list[dict[str, float | str | None]]) -> dict[str, float | str | None]:
    if not points:
        return {}
    pts = sorted(points, key=lambda x: float(x["gflops"] or -1), reverse=True)
    best = pts[0]
    gvals = [float(p["gflops"]) for p in pts if p["gflops"] is not None]
    best_g = gvals[0] if gvals else None
    second_g = gvals[1] if len(gvals) > 1 else None
    median_g = _median(gvals)
    near95 = sum(1 for g in gvals if best_g is not None and g >= 0.95 * best_g)
    out = dict(best)
    out["grid_points"] = float(len(gvals))
    out["near95_count"] = float(near95)
    out["best_over_median"] = (best_g / median_g) if best_g and median_g else None
    out["best_over_second"] = (best_g / second_g) if best_g and second_g else None
    return out


def classify_row(row: dict[str, str], baseline: str, best: dict[str, float | str | None]) -> dict[str, float | str | int | None]:
    bsmr = _f(row.get("BSMR"))
    base = _f(row.get(baseline))
    speedup = (bsmr / base) if bsmr and base else None
    dense_new = best.get("dense_block")
    dense_old = best.get("orig_dense_block")
    dens_new = best.get("dense_density")
    dens_old = best.get("orig_dense_density")
    dense_tb = best.get("dense_tb")
    sparse_tb = best.get("sparse_tb")
    dense_data = best.get("dense_data")
    sparse_data = best.get("sparse_data")
    tb_share = None
    if isinstance(dense_tb, float) and isinstance(sparse_tb, float) and dense_tb + sparse_tb > 0:
        tb_share = dense_tb / (dense_tb + sparse_tb)
    data_share = None
    if isinstance(dense_data, float) and isinstance(sparse_data, float) and dense_data + sparse_data > 0:
        data_share = dense_data / (dense_data + sparse_data)

    dense_gain = None
    if isinstance(dense_new, float) and isinstance(dense_old, float):
        dense_gain = dense_new - dense_old
    density_gain = None
    if isinstance(dens_new, float) and isinstance(dens_old, float):
        density_gain = dens_new - dens_old

    outcome = "tie"
    if bsmr is not None and base is not None:
        if bsmr < base:
            outcome = "failure"
        elif bsmr > base:
            outcome = "success"

    failure_margin = (base / bsmr) if outcome == "failure" and bsmr and base else None
    success_margin = (bsmr / base) if outcome == "success" and bsmr and base else None

    high_dense_but_fail = (
        1
        if outcome == "failure"
        and ((tb_share is not None and tb_share >= 0.30) or (data_share is not None and data_share >= 0.50))
        else 0
    )

    return {
        "file": row["file"],
        "matrix": _matrix_name(row["file"]),
        "family": _family_name(row["file"]),
        "baseline": baseline,
        "BSMR": bsmr,
        "baseline_gflops": base,
        "outcome": outcome,
        "speedup_over_baseline": speedup,
        "speedup_bin": _speedup_bin(speedup),
        "failure_margin": failure_margin,
        "success_margin": success_margin,
        "sddmm_ms": best.get("sddmm_ms"),
        "alpha": best.get("alpha"),
        "delta": best.get("delta"),
        "grid_points": best.get("grid_points"),
        "near95_count": best.get("near95_count"),
        "best_over_median": best.get("best_over_median"),
        "best_over_second": best.get("best_over_second"),
        "zero_dense_block": 1 if isinstance(dense_new, float) and dense_new <= 0 else 0,
        "dense_block_not_improved": 1 if dense_gain is not None and dense_gain <= 0 else 0,
        "density_not_improved": 1 if density_gain is not None and density_gain <= 0 else 0,
        "dense_gain": dense_gain,
        "density_gain": density_gain,
        "dense_threadblock_share": tb_share,
        "dense_data_share": data_share,
        "high_dense_but_fail": high_dense_but_fail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="更深入的失败/成功案例对照分析")
    parser.add_argument("--k", type=int, nargs="*", default=None, help="分析一个或多个 K")
    parser.add_argument("--csv", type=Path, nargs="*", default=None, help="显式指定一个或多个 results_<K>.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录")
    args = parser.parse_args()

    if args.csv:
        csv_paths = [p.resolve() for p in args.csv]
    elif args.k:
        csv_paths = [_default_results_csv(k).resolve() for k in args.k]
    else:
        csv_paths = [p.resolve() for p in _discover_default_csvs()]

    out_dir = args.out_dir.resolve() if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    bsmr_dir = _default_bsmr_dir()

    per_case_rows: list[dict[str, float | str | int | None]] = []
    per_group_rows: list[dict[str, float | str | int | None]] = []
    per_bin_rows: list[dict[str, float | str | int | None]] = []
    crossk_rows: list[dict[str, float | str | int | None]] = []
    summary_lines = ["Deep Failure Analysis", "====================", ""]

    per_baseline_matrix_outcomes: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))

    for csv_path in csv_paths:
        if not csv_path.is_file():
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        k = int(float(rows[0]["K"]))
        base_cols = {"file", "M", "N", "NNZ", "Sparsity", "K", "BSMR"}
        baselines = [c for c in rows[0].keys() if c not in base_cols]
        baselines = [b for b in baselines if any((_f(r.get(b)) or 0) > 0 for r in rows)]

        grid = load_bsmr_grid_for_k(k, bsmr_dir)
        best_map = {fp: summarize_grid(pts) for fp, pts in grid.items()}

        summary_lines.append(f"K={k}")
        summary_lines.append("  baselines: " + (", ".join(baselines) if baselines else "(none)"))

        for baseline in baselines:
            classified: list[dict[str, float | str | int | None]] = []
            for row in rows:
                base = _f(row.get(baseline))
                bsmr = _f(row.get("BSMR"))
                if bsmr is None or base is None or base <= 0:
                    continue
                best = best_map.get(row["file"])
                if not best:
                    continue
                item = classify_row(row, baseline, best)
                classified.append(item)
                per_case_rows.append({"K": k, **item})
                per_baseline_matrix_outcomes[baseline][str(item["file"])].append((k, str(item["outcome"])))

            failures = [x for x in classified if x["outcome"] == "failure"]
            successes = [x for x in classified if x["outcome"] == "success"]
            bin_order = ["<1.0x", "1.0~1.5x", "1.5~2.0x", ">=2.0x"]

            def med(key: str, items: list[dict[str, float | str | int | None]]) -> float | None:
                return _median([float(v) for v in (x.get(key) for x in items) if isinstance(v, (int, float))])

            def rate(key: str, items: list[dict[str, float | str | int | None]]) -> float | None:
                if not items:
                    return None
                vals = [int(x.get(key, 0)) for x in items]
                return sum(vals) / len(vals)

            severe = sum(1 for x in failures if isinstance(x["failure_margin"], float) and x["failure_margin"] >= 1.5)
            medium = sum(1 for x in failures if isinstance(x["failure_margin"], float) and 1.1 <= x["failure_margin"] < 1.5)
            slight = sum(1 for x in failures if isinstance(x["failure_margin"], float) and 1.0 < x["failure_margin"] < 1.1)

            group_row = {
                "K": k,
                "baseline": baseline,
                "num_failures": len(failures),
                "num_successes": len(successes),
                "failure_slight": slight,
                "failure_medium": medium,
                "failure_severe": severe,
                "failure_zero_dense_rate": rate("zero_dense_block", failures),
                "success_zero_dense_rate": rate("zero_dense_block", successes),
                "failure_no_block_gain_rate": rate("dense_block_not_improved", failures),
                "success_no_block_gain_rate": rate("dense_block_not_improved", successes),
                "failure_no_density_gain_rate": rate("density_not_improved", failures),
                "success_no_density_gain_rate": rate("density_not_improved", successes),
                "failure_tb_share_median": med("dense_threadblock_share", failures),
                "success_tb_share_median": med("dense_threadblock_share", successes),
                "failure_data_share_median": med("dense_data_share", failures),
                "success_data_share_median": med("dense_data_share", successes),
                "failure_dense_gain_median": med("dense_gain", failures),
                "success_dense_gain_median": med("dense_gain", successes),
                "failure_density_gain_median": med("density_gain", failures),
                "success_density_gain_median": med("density_gain", successes),
                "failure_best_over_median_median": med("best_over_median", failures),
                "success_best_over_median_median": med("best_over_median", successes),
                "failure_near95_count_median": med("near95_count", failures),
                "success_near95_count_median": med("near95_count", successes),
                "failure_high_dense_but_fail_rate": rate("high_dense_but_fail", failures),
            }
            per_group_rows.append(group_row)

            for speedup_bin in bin_order:
                items = [x for x in classified if x.get("speedup_bin") == speedup_bin]
                if not items:
                    continue
                per_bin_rows.append({
                    "K": k,
                    "baseline": baseline,
                    "speedup_bin": speedup_bin,
                    "num_cases": len(items),
                    "case_ratio": len(items) / len(classified) if classified else None,
                    "zero_dense_rate": rate("zero_dense_block", items),
                    "no_block_gain_rate": rate("dense_block_not_improved", items),
                    "no_density_gain_rate": rate("density_not_improved", items),
                    "tb_share_median": med("dense_threadblock_share", items),
                    "data_share_median": med("dense_data_share", items),
                    "dense_gain_median": med("dense_gain", items),
                    "density_gain_median": med("density_gain", items),
                    "best_over_median_median": med("best_over_median", items),
                    "near95_count_median": med("near95_count", items),
                })

            summary_lines.extend([
                f"  {baseline}: fail={len(failures)}, success={len(successes)}",
                f"    fail: zero_dense={_fmt(group_row['failure_zero_dense_rate'])}, no_block_gain={_fmt(group_row['failure_no_block_gain_rate'])}, "
                f"no_density_gain={_fmt(group_row['failure_no_density_gain_rate'])}, tb_share={_fmt(group_row['failure_tb_share_median'])}, "
                f"data_share={_fmt(group_row['failure_data_share_median'])}, best/median={_fmt(group_row['failure_best_over_median_median'])}, near95={_fmt(group_row['failure_near95_count_median'])}",
                f"    succ: zero_dense={_fmt(group_row['success_zero_dense_rate'])}, no_block_gain={_fmt(group_row['success_no_block_gain_rate'])}, "
                f"no_density_gain={_fmt(group_row['success_no_density_gain_rate'])}, tb_share={_fmt(group_row['success_tb_share_median'])}, "
                f"data_share={_fmt(group_row['success_data_share_median'])}, best/median={_fmt(group_row['success_best_over_median_median'])}, near95={_fmt(group_row['success_near95_count_median'])}",
                f"    failure severity: slight={slight}, medium={medium}, severe={severe}",
            ])
            for speedup_bin in bin_order:
                items = [x for x in classified if x.get("speedup_bin") == speedup_bin]
                if not items:
                    continue
                summary_lines.append(
                    "    "
                    f"{speedup_bin}: n={len(items)}, zero_dense={_fmt(rate('zero_dense_block', items))}, "
                    f"no_block_gain={_fmt(rate('dense_block_not_improved', items))}, "
                    f"no_density_gain={_fmt(rate('density_not_improved', items))}, "
                    f"tb_share={_fmt(med('dense_threadblock_share', items))}, "
                    f"data_share={_fmt(med('dense_data_share', items))}"
                )
        summary_lines.append("")

    for baseline, matrix_map in per_baseline_matrix_outcomes.items():
        for file_path, outcomes in matrix_map.items():
            ks = [k for k, _ in outcomes]
            fail_k = [k for k, out in outcomes if out == "failure"]
            succ_k = [k for k, out in outcomes if out == "success"]
            crossk_rows.append({
                "baseline": baseline,
                "file": file_path,
                "matrix": _matrix_name(file_path),
                "family": _family_name(file_path),
                "observed_k_count": len(ks),
                "failure_k_count": len(fail_k),
                "success_k_count": len(succ_k),
                "failure_ks": ",".join(map(str, sorted(fail_k))),
                "success_ks": ",".join(map(str, sorted(succ_k))),
                "always_fail": 1 if len(fail_k) == len(ks) and len(ks) > 0 else 0,
                "always_success": 1 if len(succ_k) == len(ks) and len(ks) > 0 else 0,
            })

    def write_csv(path: Path, rows: list[dict[str, float | str | int | None]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out_dir / "per_case_diagnostics.csv", per_case_rows)
    write_csv(out_dir / "per_group_summary.csv", per_group_rows)
    write_csv(out_dir / "per_bin_summary.csv", per_bin_rows)
    write_csv(out_dir / "cross_k_consistency.csv", crossk_rows)

    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"分析完成，结果已输出到: {out_dir}")


if __name__ == "__main__":
    main()
