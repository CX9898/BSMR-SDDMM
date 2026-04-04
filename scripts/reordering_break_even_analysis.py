#!/usr/bin/env python3
"""
计算 BSMR 重排序开销的摊销轮数。

这里不和任何基线比较，只回答一个问题：
  需要做多少轮 BSMR SDDMM，才能分担掉一次重排序的耗时？

定义:
  amortization_rounds = reordering_time_ms / bsmr_sddmm_time_ms

输入:
  1. 结果 CSV（如 results_128.csv），其中 BSMR 列为 GFLOPS
  2. 同级目录下的 BSMR_results/BSMR_k_<K>_a_*_d_*.log

输出:
  - 一个按矩阵行数分组的中位数汇总 CSV，包含:
      * row_range
      * median_amortization_rounds
      * median_reordering_ms
      * median_bsmr_sddmm_ms
  - 终端摘要统计
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


BSMR_LOG_NAME_RE = re.compile(r"BSMR_k_(\d+)_a_([\d.]+)_d_([\d.]+)\.log$", re.IGNORECASE)
ROW_GROUP_SIZE = 10000


def _f(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _matrix_name(path: str) -> str:
    base = path.rstrip("/").split("/")[-1]
    return base[:-4] if base.endswith(".mtx") else base


def _time_ms_from_gflops(nnz: float | None, k: float | None, gflops: float | None) -> float | None:
    if nnz is None or k is None or gflops is None or gflops <= 0:
        return None
    # SDDMM 近似 FLOPs: 2 * NNZ * K；GFLOPS = 1e9 flop/s
    # 换算到毫秒后: time_ms = flops / gflops / 1e6
    return (2.0 * nnz * k) / (gflops * 1e6)


def _format_float(v: float | None, digits: int = 6) -> str:
    if v is None:
        return ""
    if math.isinf(v):
        return "inf"
    return f"{v:.{digits}f}"


def build_best_bsmr_map(bsmr_dir: Path, k: int) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    if not bsmr_dir.is_dir():
        return best

    for path in sorted(bsmr_dir.glob(f"BSMR_k_{k}_a_*_d_*.log")):
        match = BSMR_LOG_NAME_RE.search(path.name)
        if not match or int(match.group(1)) != k:
            continue

        alpha = match.group(2)
        delta = match.group(3)
        text = path.read_text(encoding="utf-8", errors="replace")
        for block in re.split(r"(?:^|\n)---New data---\n", text):
            if "[File :" not in block:
                continue

            fm = re.search(r"\[File : ([^\]]+)\]", block)
            gm = re.search(r"\[bsmr_gflops : ([^\]]+)\]", block)
            rm = re.search(r"\[bsmr_reordering : ([^\]]+)\]", block)
            sm = re.search(r"\[bsmr_sddmm : ([^\]]+)\]", block)
            if not fm or not gm:
                continue

            file_path = fm.group(1).strip()
            gflops = _f(gm.group(1))
            if gflops is None:
                continue

            prev = best.get(file_path)
            prev_gflops = _f(prev.get("bsmr_gflops_log")) if prev else None
            if prev_gflops is not None and gflops < prev_gflops - 1e-9:
                continue

            best[file_path] = {
                "bsmr_alpha": alpha,
                "bsmr_delta": delta,
                "bsmr_log_basename": path.name,
                "bsmr_gflops_log": str(gflops),
                "bsmr_reordering_ms": rm.group(1).strip() if rm else "",
                "bsmr_sddmm_ms_log": sm.group(1).strip() if sm else "",
            }
    return best


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "count": float(len(values)),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": sorted(values)[max(0, math.ceil(len(values) * 0.9) - 1)],
        "min": min(values),
        "max": max(values),
    }


def _row_group_label(m: float) -> str:
    lo = int(m // ROW_GROUP_SIZE) * ROW_GROUP_SIZE
    hi = lo + ROW_GROUP_SIZE
    return f"[{lo}, {hi})"


def analyze(csv_path: Path, output_csv: Path, bsmr_dir: Path | None = None) -> None:
    rows = read_rows(csv_path)
    if not rows:
        raise SystemExit(f"输入 CSV 为空: {csv_path}")

    first_k = _f(rows[0].get("K"))
    if first_k is None:
        raise SystemExit("CSV 中缺少 K 列，无法定位对应 BSMR 日志。")
    k = int(first_k)

    if bsmr_dir is None:
        bsmr_dir = csv_path.parent.parent / "BSMR_results"
    best_bsmr = build_best_bsmr_map(bsmr_dir, k)

    amortization_rounds_list: list[float] = []
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"amortization_rounds": [], "reordering_ms": [], "bsmr_sddmm_ms": []}
    )
    matched_logs = 0

    for row in rows:
        file_path = row.get("file", "")
        m_val = _f(row.get("M"))
        nnz = _f(row.get("NNZ"))
        k_val = _f(row.get("K"))
        bsmr_gflops = _f(row.get("BSMR"))
        bsmr_time_ms = _time_ms_from_gflops(nnz, k_val, bsmr_gflops)

        log_info = best_bsmr.get(file_path, {})
        reorder_ms = _f(log_info.get("bsmr_reordering_ms"))
        bsmr_time_ms_log = _f(log_info.get("bsmr_sddmm_ms_log"))
        if log_info:
            matched_logs += 1

        amortization_rounds = None
        if reorder_ms is not None and bsmr_time_ms is not None and bsmr_time_ms > 0:
            amortization_rounds = reorder_ms / bsmr_time_ms

        if amortization_rounds is not None and not math.isinf(amortization_rounds):
            amortization_rounds_list.append(amortization_rounds)
            if m_val is not None:
                label = _row_group_label(m_val)
                grouped[label]["amortization_rounds"].append(amortization_rounds)
                if reorder_ms is not None:
                    grouped[label]["reordering_ms"].append(reorder_ms)
                if bsmr_time_ms_log is not None:
                    grouped[label]["bsmr_sddmm_ms"].append(bsmr_time_ms_log)

    output_rows: list[dict[str, str]] = []
    for label in sorted(grouped, key=lambda x: int(x.split(",")[0][1:])):
        vals = grouped[label]
        if not vals["amortization_rounds"]:
            continue
        output_rows.append(
            {
                "row_range": label,
                "median_amortization_rounds": _format_float(statistics.median(vals["amortization_rounds"]), 2),
                "median_reordering_ms": _format_float(statistics.median(vals["reordering_ms"]), 2),
                "median_bsmr_sddmm_ms": _format_float(statistics.median(vals["bsmr_sddmm_ms"]), 4),
            }
        )

    fieldnames = [
        "row_range",
        "median_amortization_rounds",
        "median_reordering_ms",
        "median_bsmr_sddmm_ms",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"输入 CSV: {csv_path}")
    print(f"匹配到 BSMR 日志: {matched_logs}/{len(rows)}")
    print(f"输出结果: {output_csv}")
    print()
    stats = summarize(amortization_rounds_list)
    if stats is None:
        print("摊销轮数统计: 无可用样本")
    else:
        print(
            "摊销轮数统计: "
            f"count={int(stats['count'])}, mean={stats['mean']:.2f}, "
            f"median={stats['median']:.2f}, p90={stats['p90']:.2f}, "
            f"min={stats['min']:.2f}, max={stats['max']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="按矩阵行数分组统计 BSMR 重排序摊销轮数中位数")
    parser.add_argument("--csv", required=True, help="结果 CSV 路径，例如 scripts/results_suiteSparse_dataset/k128/results_128.csv")
    parser.add_argument("--bsmr-dir", help="BSMR 日志目录，默认自动推断为 CSV 上两级目录下的 BSMR_results")
    parser.add_argument("--out", help="输出 CSV 路径，默认写到输入 CSV 同目录")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.is_file():
        raise SystemExit(f"找不到输入 CSV: {csv_path}")

    bsmr_dir = Path(args.bsmr_dir).resolve() if args.bsmr_dir else None
    default_out = csv_path.parent / f"{csv_path.stem}_reordering_break_even_row_groups_median.csv"
    out_path = Path(args.out).resolve() if args.out else default_out

    analyze(csv_path=csv_path, output_csv=out_path, bsmr_dir=bsmr_dir)


if __name__ == "__main__":
    main()
