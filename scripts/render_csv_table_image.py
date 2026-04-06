#!/usr/bin/env python3
"""
将 CSV 渲染成适合论文使用的表格图片（PNG/PDF）。
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"输入 CSV 为空: {path}")
    return rows[0], rows[1:]


def wrap_cell(text: str, width: int) -> str:
    if not text:
        return ""
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def prettify_header(text: str) -> str:
    mapping = {
        "Range of max(M, N)": "max(M,N) range",
        "# Matrices": "# matrices",
        "Median Row Reordering Time (ms)": "Median row reorder (ms)",
        "Median Column Reordering Time (ms)": "Median col reorder (ms)",
        "Median Total Reordering Time (ms)": "Median total (ms)",
        "Median BSMR SDDMM Time (ms)": "Median BSMR SDDMM (ms)",
        "Median Break-even Rounds": "Median break-even rounds",
    }
    return mapping.get(text, text)


def infer_col_widths(headers: list[str], rows: list[list[str]]) -> list[float]:
    # For the paper table, distribute widths according to header text only.
    # This keeps the layout aligned with column semantics instead of being
    # dominated by long numeric contents.
    pretty_headers = [prettify_header(h).replace("\n", " ") for h in headers]
    header_lens: list[int] = [max(6, len(h)) for h in pretty_headers]

    widths: list[float] = []
    for i, m in enumerate(header_lens):
        if i == 0:
            widths.append(max(0.10, min(0.18, m / 55.0)))
        elif i == 1:
            widths.append(max(0.06, min(0.11, m / 70.0)))
        else:
            widths.append(max(0.10, min(0.19, m / 60.0)))

    total = sum(widths)
    return [w / total for w in widths]


def render_table(
    headers: list[str],
    rows: list[list[str]],
    output_png: Path | None,
    output_pdf: Path | None,
    title: str | None,
) -> None:
    wrapped_headers = [prettify_header(h) for h in headers]
    wrapped_rows = [list(row) for row in rows]

    col_widths = infer_col_widths(headers, rows)
    n_rows = len(rows) + 1
    fig_h = max(2.8, 0.32 * n_rows + (0.40 if title else 0.10))
    fig_w = 15.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, pad=14)

    table = ax.table(
        cellText=wrapped_rows,
        colLabels=wrapped_headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[0.01, 0.02, 0.98, 0.96],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.35)

    header_color = "#D9E6F2"
    alt_row_color = "#F7F9FC"
    edge_color = "#4A4A4A"

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.6)
        cell.PAD = 0.02
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", va="center")
            cell.set_height(cell.get_height() * 1.10)
        else:
            if r % 2 == 0:
                cell.set_facecolor(alt_row_color)
        if c == 0:
            cell.set_text_props(ha="left")

    plt.tight_layout(pad=0.2)

    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def default_outputs(csv_path: Path) -> tuple[Path, Path]:
    stem = csv_path.with_suffix("")
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="将 CSV 渲染为表格图片（PNG/PDF）。")
    parser.add_argument("--csv", required=True, type=Path, help="输入 CSV 路径")
    parser.add_argument("--png", type=Path, default=None, help="输出 PNG 路径")
    parser.add_argument("--pdf", type=Path, default=None, help="输出 PDF 路径")
    parser.add_argument("--title", default=None, help="可选标题")
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    headers, rows = read_csv(csv_path)
    default_png, default_pdf = default_outputs(csv_path)
    out_png = args.png.resolve() if args.png else default_png
    out_pdf = args.pdf.resolve() if args.pdf else default_pdf

    render_table(headers, rows, out_png, out_pdf, args.title)

    print(f"输入 CSV: {csv_path}")
    print(f"PNG: {out_png}")
    print(f"PDF: {out_pdf}")


if __name__ == "__main__":
    main()
