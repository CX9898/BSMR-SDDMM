#!/usr/bin/env python3
"""
按矩阵规模分组统计 BSMR 重排序相关时间。

输入:
  extract_bsmr_reordering_details.py 生成的明细 CSV

输出:
  每个分组内的样本数，以及以下指标的均值/中位数:
    - 行重排序时间
    - 列重排序时间
    - 总重排序时间
    - BSMR SDDMM 时间
    - 盈亏平衡轮数 = 总重排序时间 / BSMR SDDMM 时间
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


VALUE_COLUMNS = [
    "bsmr_rowReordering_ms",
    "bsmr_colReordering_ms",
    "bsmr_reordering_ms",
    "bsmr_sddmm_ms",
]

SHARE_COLUMNS = [
    "row_share_pct",
    "col_share_pct",
]


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


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}%"


def _fmt_pct_latex(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}\\%"


def _ceil_int_str(value: str | None) -> str:
    fv = _f(value)
    if fv is None:
        return ""
    return str(int(math.ceil(fv)))


def _fmt_num_for_latex(value: str, digits: int) -> str:
    fv = _f(value)
    if fv is None:
        return value
    return f"{fv:.{digits}f}"


def _group_bounds(group_val: float, min_value: int, group_size: int) -> tuple[int, int] | None:
    if group_val < min_value:
        return None
    idx = int((group_val - min_value) // group_size)
    lo = min_value + idx * group_size
    hi = lo + group_size
    return lo, hi


def _group_label(lo: int, hi: int) -> str:
    return f"[{lo}, {hi})"


def _paper_group_header(group_by: str) -> str:
    if group_by == "m":
        return "Row Count Range"
    if group_by == "max_mn":
        return "Range of max(M, N)"
    raise ValueError(f"unsupported group_by: {group_by}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_values(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.median(values)


def _group_value(row: dict[str, str], group_by: str) -> float | None:
    m_val = _f(row.get("M"))
    n_val = _f(row.get("N"))
    if group_by == "m":
        return m_val
    if m_val is None or n_val is None:
        return None
    if group_by == "max_mn":
        return max(m_val, n_val)
    raise ValueError(f"unsupported group_by: {group_by}")


def analyze(rows: list[dict[str, str]], min_value: int, group_size: int, group_by: str) -> list[dict[str, str]]:
    grouped: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: {
            "bsmr_rowReordering_ms": [],
            "bsmr_colReordering_ms": [],
            "bsmr_reordering_ms": [],
            "bsmr_sddmm_ms": [],
            "break_even_rounds": [],
            "row_share_pct": [],
            "col_share_pct": [],
        }
    )
    group_counts: dict[tuple[int, int], int] = defaultdict(int)

    for row in rows:
        group_val = _group_value(row, group_by=group_by)
        if group_val is None:
            continue
        bounds = _group_bounds(group_val, min_value=min_value, group_size=group_size)
        if bounds is None:
            continue

        group_counts[bounds] += 1
        bucket = grouped[bounds]
        for key in VALUE_COLUMNS:
            val = _f(row.get(key))
            if val is not None:
                bucket[key].append(val)

        reorder_ms = _f(row.get("bsmr_reordering_ms"))
        row_reorder_ms = _f(row.get("bsmr_rowReordering_ms"))
        col_reorder_ms = _f(row.get("bsmr_colReordering_ms"))
        sddmm_ms = _f(row.get("bsmr_sddmm_ms"))
        if reorder_ms is not None and reorder_ms > 0:
            if row_reorder_ms is not None:
                bucket["row_share_pct"].append(100.0 * row_reorder_ms / reorder_ms)
            if col_reorder_ms is not None:
                bucket["col_share_pct"].append(100.0 * col_reorder_ms / reorder_ms)
        if reorder_ms is not None and sddmm_ms is not None and sddmm_ms > 0:
            rounds = reorder_ms / sddmm_ms
            if not math.isinf(rounds):
                bucket["break_even_rounds"].append(rounds)

    output: list[dict[str, str]] = []
    for bounds in sorted(grouped):
        lo, hi = bounds
        vals = grouped[bounds]
        row_out: dict[str, str] = {
            "group_by": group_by,
            "group_range": _group_label(lo, hi),
            "count": str(group_counts[bounds]),
        }

        for key in VALUE_COLUMNS + ["break_even_rounds"] + SHARE_COLUMNS:
            mean_v, median_v = summarize_values(vals[key])
            row_out[f"mean_{key}"] = _fmt(mean_v, 4)
            row_out[f"median_{key}"] = _fmt(median_v, 4)

        output.append(row_out)
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "group_by",
        "group_range",
        "count",
        "mean_bsmr_rowReordering_ms",
        "median_bsmr_rowReordering_ms",
        "mean_bsmr_colReordering_ms",
        "median_bsmr_colReordering_ms",
        "mean_bsmr_reordering_ms",
        "median_bsmr_reordering_ms",
        "mean_bsmr_sddmm_ms",
        "median_bsmr_sddmm_ms",
        "mean_break_even_rounds",
        "median_break_even_rounds",
        "mean_row_share_pct",
        "median_row_share_pct",
        "mean_col_share_pct",
        "median_col_share_pct",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_paper_csv(path: Path, rows: list[dict[str, str]], group_by: str) -> None:
    group_header = _paper_group_header(group_by)
    fieldnames = [
        group_header,
        "# Matrices",
        "Median Row Reordering Time (ms)",
        "Median Column Reordering Time (ms)",
        "Median Total Reordering Time (ms)",
        "Median BSMR SDDMM Time (ms)",
        "Median Break-even Rounds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_time = row["median_bsmr_rowReordering_ms"]
            col_time = row["median_bsmr_colReordering_ms"]
            row_share = row["median_row_share_pct"]
            col_share = row["median_col_share_pct"]
            writer.writerow(
                {
                    group_header: row["group_range"],
                    "# Matrices": row["count"],
                    "Median Row Reordering Time (ms)": f"{row_time} ({_fmt_pct(_f(row_share))})" if row_time else "",
                    "Median Column Reordering Time (ms)": f"{col_time} ({_fmt_pct(_f(col_share))})" if col_time else "",
                    "Median Total Reordering Time (ms)": row["median_bsmr_reordering_ms"],
                    "Median BSMR SDDMM Time (ms)": row["median_bsmr_sddmm_ms"],
                    "Median Break-even Rounds": _ceil_int_str(row["median_break_even_rounds"]),
                }
            )


def write_paper_tex(path: Path, rows: list[dict[str, str]], group_by: str) -> None:
    group_header = "max(M,N) range" if group_by == "max_mn" else "Row count range"
    caption = (
        "Break-even analysis of BSMR reordering overhead in SDDMM on RTX 4090 GPU (K=128). "
        "The first column lists the matrix-size groups. Row and column entries show median reordering time with "
        "their share of total reordering time in parentheses."
    )
    label = "tab:k128_reordering_break_even_by_maxmn" if group_by == "max_mn" else "tab:k128_reordering_break_even_by_rows"

    lines: list[str] = [
        r"\begin{table}[h]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lcccccc}",
        r"\toprule",
        rf"\textbf{{{group_header}}} & \textbf{{\#Matrices}} & \textbf{{Row (share)}} & "
        r"\textbf{Col (share)} & \textbf{Total (ms)} & \textbf{BSMR (ms)} & \textbf{Break-even} \\",
        r"\midrule",
    ]

    for row in rows:
        row_range = "{" + row["group_range"] + "}"
        count = row["count"]
        row_time = row["median_bsmr_rowReordering_ms"]
        col_time = row["median_bsmr_colReordering_ms"]
        total_time = row["median_bsmr_reordering_ms"]
        sddmm_time = row["median_bsmr_sddmm_ms"]
        break_even = row["median_break_even_rounds"]
        row_share = _fmt_pct_latex(_f(row["median_row_share_pct"]))
        col_share = _fmt_pct_latex(_f(row["median_col_share_pct"]))

        row_cell = f"{_fmt_num_for_latex(row_time, 2)} ({row_share})"
        col_cell = f"{_fmt_num_for_latex(col_time, 2)} ({col_share})"
        total_cell = _fmt_num_for_latex(total_time, 2)
        sddmm_cell = _fmt_num_for_latex(sddmm_time, 3)
        break_even_cell = _ceil_int_str(break_even)

        lines.append(
            f"{row_range} & {count} & {row_cell} & {col_cell} & {total_cell} & {sddmm_cell} & {break_even_cell} \\\\"
        )

    lines.extend(
        [
            r"\botrule",
            r"\end{tabular*}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="按矩阵规模分组汇总 BSMR 重排序与盈亏平衡轮数。")
    parser.add_argument("--csv", required=True, type=Path, help="输入明细 CSV，例如 results_128_bsmr_reordering_details.csv")
    parser.add_argument("--out", type=Path, default=None, help="输出汇总 CSV 路径")
    parser.add_argument("--min-row", type=int, default=10000, help="分组起始值，默认 10000")
    parser.add_argument("--group-size", type=int, default=10000, help="分组大小，默认 10000")
    parser.add_argument(
        "--group-by",
        choices=["m", "max_mn"],
        default="m",
        help="分组依据: m 表示按 M 分组, max_mn 表示按 max(M, N) 分组",
    )
    parser.add_argument("--paper-ready", action="store_true", help="额外输出一份适合论文表格展示的精简 CSV")
    parser.add_argument("--latex-ready", action="store_true", help="额外输出一份可直接用于论文的 LaTeX 表格片段")
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    rows = read_rows(csv_path)
    if not rows:
        raise SystemExit(f"输入 CSV 为空: {csv_path}")

    default_name = f"{csv_path.stem}_{args.group_by}_group_summary.csv"
    out_path = args.out.resolve() if args.out else csv_path.parent / default_name
    summary_rows = analyze(rows, min_value=args.min_row, group_size=args.group_size, group_by=args.group_by)
    write_csv(out_path, summary_rows)
    paper_out_path = None
    latex_out_path = None
    if args.paper_ready:
        paper_out_path = out_path.with_name(f"{out_path.stem}_paper.csv")
        write_paper_csv(paper_out_path, summary_rows, group_by=args.group_by)
    if args.latex_ready:
        latex_out_path = out_path.with_name(f"{out_path.stem}_paper.tex")
        write_paper_tex(latex_out_path, summary_rows, group_by=args.group_by)

    print(f"输入 CSV: {csv_path}")
    print(f"输出 CSV: {out_path}")
    if paper_out_path is not None:
        print(f"论文表格 CSV: {paper_out_path}")
    if latex_out_path is not None:
        print(f"LaTeX 表格: {latex_out_path}")
    print(f"分组依据: {args.group_by}")
    print(f"分组起点: {args.min_row}")
    print(f"分组大小: {args.group_size}")
    print(f"输出分组数: {len(summary_rows)}")


if __name__ == "__main__":
    main()
