#!/usr/bin/env python3
"""
从 results_*.csv 中提取 BSMR 的详细日志信息，并导出为新的 CSV。

用途:
  - 原始 results_*.csv 只保留了汇总后的 BSMR 指标
  - 本脚本会到同级目录下的 BSMR_results/BSMR_k_<K>_a_*_d_*.log 中，
    按矩阵文件和 BSMR GFLOPS 去匹配对应记录
  - 导出 alpha / delta / 重排序时间 / block 统计等详细字段

示例:
  python3 scripts/extract_bsmr_reordering_details.py \
      --csv scripts/results_suiteSparse_dataset/k128/results_128.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


LOG_NAME_RE = re.compile(r"BSMR_k_(\d+)_a_([\d.]+)_d_([\d.]+)\.log$", re.IGNORECASE)

BLOCK_PATTERNS = {
    "file": r"\[File : ([^\]]+)\]",
    "bsmr_alpha": r"\[bsmr_alpha : ([^\]]+)\]",
    "bsmr_delta": r"\[bsmr_delta : ([^\]]+)\]",
    "bsmr_gflops_log": r"\[bsmr_gflops : ([^\]]+)\]",
    "bsmr_numClusters": r"\[bsmr_numClusters : ([^\]]+)\]",
    "original_numDenseBlock": r"\[original_numDenseBlock : ([^\]]+)\]",
    "original_averageDensity": r"\[original_averageDensity : ([^\]]+)\]",
    "bsmr_numDenseBlock": r"\[bsmr_numDenseBlock : ([^\]]+)\]",
    "bsmr_averageDensity": r"\[bsmr_averageDensity : ([^\]]+)\]",
    "bsmr_rowReordering_ms": r"\[bsmr_rowReordering : ([^\]]+)\]",
    "bsmr_colReordering_ms": r"\[bsmr_colReordering : ([^\]]+)\]",
    "bsmr_reordering_ms": r"\[bsmr_reordering : ([^\]]+)\]",
    "bsmr_sddmm_ms": r"\[bsmr_sddmm : ([^\]]+)\]",
    "bsmr_numDenseThreadBlocks": r"\[bsmr_numDenseThreadBlocks : ([^\]]+)\]",
    "bsmr_numSparseThreadBlocks": r"\[bsmr_numSparseThreadBlocks : ([^\]]+)\]",
    "bsmr_threadBlockRatio": r"\[bsmr_threadBlockRatio : ([^\]]+)\]",
    "bsmr_numDenseData": r"\[bsmr_numDenseData : ([^\]]+)\]",
    "bsmr_numSparseData": r"\[bsmr_numSparseData : ([^\]]+)\]",
    "bsmr_dataRatio": r"\[bsmr_dataRatio: ([^\]]+)\]",
}

EXTRA_COLUMNS = [
    "matrix",
    "bsmr_log_basename",
    "bsmr_candidate_count",
    "bsmr_match_status",
    "bsmr_gflops_log",
    "bsmr_gflops_abs_diff",
    "bsmr_alpha",
    "bsmr_delta",
    "bsmr_numClusters",
    "original_numDenseBlock",
    "original_averageDensity",
    "bsmr_numDenseBlock",
    "bsmr_averageDensity",
    "bsmr_rowReordering_ms",
    "bsmr_colReordering_ms",
    "bsmr_reordering_ms",
    "bsmr_sddmm_ms",
    "bsmr_numDenseThreadBlocks",
    "bsmr_numSparseThreadBlocks",
    "bsmr_threadBlockRatio",
    "bsmr_numDenseData",
    "bsmr_numSparseData",
    "bsmr_dataRatio",
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


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def _matrix_name(path: str) -> str:
    base = path.rstrip("/").split("/")[-1]
    return base[:-4] if base.endswith(".mtx") else base


def _infer_k(csv_path: Path, rows: list[dict[str, str]]) -> int:
    if rows:
        k_val = _f(rows[0].get("K"))
        if k_val is not None:
            return int(k_val)
    match = re.search(r"results_(\d+)\.csv$", csv_path.name)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"k(\d+)", csv_path.parent.name)
    if match:
        return int(match.group(1))
    raise SystemExit(f"无法从 CSV 推断 K: {csv_path}")


def _default_bsmr_dir(csv_path: Path) -> Path:
    return csv_path.parent.parent / "BSMR_results"


def _default_output(csv_path: Path) -> Path:
    return csv_path.parent / f"{csv_path.stem}_bsmr_reordering_details.csv"


def _match_tolerance(csv_val: float | None, log_val: float | None) -> float:
    if csv_val is None or log_val is None:
        return 0.0
    return 0.5 + 1e-4 * max(abs(csv_val), abs(log_val))


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def parse_bsmr_logs(bsmr_dir: Path, k: int) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for log_path in sorted(bsmr_dir.glob(f"BSMR_k_{k}_a_*_d_*.log")):
        match = LOG_NAME_RE.search(log_path.name)
        if not match or int(match.group(1)) != k:
            continue

        text = log_path.read_text(encoding="utf-8", errors="replace")
        for block in re.split(r"(?:^|\n)---New data---\n", text):
            if "[File :" not in block:
                continue

            row: dict[str, str] = {"bsmr_log_basename": log_path.name}
            ok = True
            for key, pattern in BLOCK_PATTERNS.items():
                mm = re.search(pattern, block)
                if key == "file" and not mm:
                    ok = False
                    break
                row[key] = mm.group(1).strip() if mm else ""
            if not ok:
                continue
            grouped.setdefault(row["file"], []).append(row)
    return grouped


def choose_best_candidate(candidates: list[dict[str, str]], csv_bsmr: float | None) -> dict[str, str] | None:
    if not candidates:
        return None

    def sort_key(item: dict[str, str]) -> tuple[float, float, float, float]:
        log_gflops = _f(item.get("bsmr_gflops_log"))
        if csv_bsmr is None or log_gflops is None:
            diff = float("inf")
        else:
            diff = abs(csv_bsmr - log_gflops)
        reorder_ms = _f(item.get("bsmr_reordering_ms"))
        return (
            diff,
            0.0 if log_gflops is not None else 1.0,
            -(log_gflops if log_gflops is not None else float("-inf")),
            reorder_ms if reorder_ms is not None else float("inf"),
        )

    if csv_bsmr is None:
        return max(candidates, key=lambda x: _f(x.get("bsmr_gflops_log")) or float("-inf"))
    return min(candidates, key=sort_key)


def enrich_rows(rows: list[dict[str, str]], log_map: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for row in rows:
        out = dict(row)
        file_path = row.get("file", "")
        out["matrix"] = _matrix_name(file_path)

        candidates = log_map.get(file_path, [])
        out["bsmr_candidate_count"] = str(len(candidates))
        csv_bsmr = _f(row.get("BSMR"))
        best = choose_best_candidate(candidates, csv_bsmr)

        for col in EXTRA_COLUMNS:
            out.setdefault(col, "")

        if best is None:
            out["bsmr_match_status"] = "no_log"
            output_rows.append(out)
            continue

        for key, value in best.items():
            if key != "file":
                out[key] = value

        log_bsmr = _f(best.get("bsmr_gflops_log"))
        diff = None if csv_bsmr is None or log_bsmr is None else abs(csv_bsmr - log_bsmr)
        out["bsmr_gflops_abs_diff"] = _fmt_float(diff)

        if csv_bsmr is None:
            out["bsmr_match_status"] = "missing_csv_bsmr"
        elif log_bsmr is None:
            out["bsmr_match_status"] = "missing_log_bsmr"
        elif diff is not None and diff <= _match_tolerance(csv_bsmr, log_bsmr):
            out["bsmr_match_status"] = "ok"
        else:
            out["bsmr_match_status"] = "closest_mismatch"

        output_rows.append(out)
    return output_rows


def write_csv(fieldnames: list[str], rows: list[dict[str, str]], output_csv: Path) -> None:
    final_fields = list(fieldnames)
    for col in EXTRA_COLUMNS:
        if col not in final_fields:
            final_fields.append(col)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="提取 BSMR 重排序详细信息到 CSV。")
    parser.add_argument("--csv", required=True, type=Path, help="输入结果 CSV，例如 scripts/results_suiteSparse_dataset/k128/results_128.csv")
    parser.add_argument("--bsmr-dir", type=Path, default=None, help="BSMR_results 目录，默认自动推断")
    parser.add_argument("--out", type=Path, default=None, help="输出 CSV 路径，默认写到输入 CSV 同目录")
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    fieldnames, rows = read_csv_rows(csv_path)
    if not rows:
        raise SystemExit(f"输入 CSV 为空: {csv_path}")

    k = _infer_k(csv_path, rows)
    bsmr_dir = args.bsmr_dir.resolve() if args.bsmr_dir else _default_bsmr_dir(csv_path)
    output_csv = args.out.resolve() if args.out else _default_output(csv_path)

    if not bsmr_dir.is_dir():
        raise SystemExit(f"BSMR 日志目录不存在: {bsmr_dir}")

    log_map = parse_bsmr_logs(bsmr_dir, k)
    enriched = enrich_rows(rows, log_map)
    write_csv(fieldnames, enriched, output_csv)

    ok_count = sum(1 for r in enriched if r.get("bsmr_match_status") == "ok")
    mismatch_count = sum(1 for r in enriched if r.get("bsmr_match_status") == "closest_mismatch")
    no_log_count = sum(1 for r in enriched if r.get("bsmr_match_status") == "no_log")

    print(f"输入 CSV: {csv_path}")
    print(f"K: {k}")
    print(f"BSMR 日志目录: {bsmr_dir}")
    print(f"输出 CSV: {output_csv}")
    print(f"总行数: {len(enriched)}")
    print(f"匹配成功: {ok_count}")
    print(f"近似匹配但 GFLOPS 不一致: {mismatch_count}")
    print(f"未找到日志: {no_log_count}")


if __name__ == "__main__":
    main()
