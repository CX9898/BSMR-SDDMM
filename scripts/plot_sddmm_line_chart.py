import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import matplotlib
from matplotlib.ticker import LogLocator

# ✅ 设置全局样式（科研级别图表风格）
matplotlib.rcParams.update({
    'figure.figsize': (12, 6),
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 6
})


def parse_markdown_data(file_path):
    data = []
    current_file = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            file_match = re.search(r'file\s*:\s*.*/(.+\.mtx)', line)
            if file_match:
                current_file = file_match.group(1).strip()
            elif line.strip().startswith("|") and not line.strip().startswith("| M ") and current_file:
                parts = [x.strip().replace('%', '') for x in line.strip().split("|")[1:-1]]
                if len(parts) >= 9:
                    try:
                        row = {
                            "file": current_file,
                            "NNZ": int(parts[2]),
                            "K": int(parts[4]),
                            "bsmr_gflops": float(parts[5]) if parts[5] else None,
                            "cuSDDMM_gflops": float(parts[6]) if parts[6] else None,
                            "cuSparse_gflops": float(parts[7]) if parts[7] else None,
                            "ASpT_gflops": float(parts[8]) if parts[8] else None
                        }
                        data.append(row)
                    except ValueError:
                        continue
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="测试结果可视化（折线图）")
    parser.add_argument('-file', type=str, required=True, help="输入的 Markdown 结果文件路径")
    parser.add_argument('-outdir', type=str, default='.', help="图表输出目录（默认当前目录）")
    args = parser.parse_args()

    df = parse_markdown_data(args.file)
    df = df.sort_values(by="NNZ").reset_index(drop=True)
    df = df[(df["NNZ"] >= 1000) & (df["NNZ"] <= 20000)]
    df = df.dropna(subset=["K", "bsmr_gflops"])
    df = df.drop_duplicates(subset=["file", "K"])

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_K = sorted(df["K"].unique())

    for k in unique_K:
        subset = df[df["K"] == k].copy().reset_index(drop=True)
        subset = subset.iloc[::3].copy().reset_index(drop=True)  # 子采样

        x = subset["NNZ"].values

        fig, ax = plt.subplots()
        ax.plot(x, subset["cuSDDMM_gflops"], marker='s', label="cuSDDMM", alpha=0.7)
        ax.plot(x, subset["cuSparse_gflops"], marker='o', label="cuSparse", alpha=0.7)
        ax.plot(x, subset["bsmr_gflops"], marker='^', label="BSMR", alpha=0.7)
        ax.plot(x, subset["ASpT_gflops"], marker='d', label="ASpT", alpha=0.7)

        ax.set_title(f"K={k}")
        ax.set_ylabel("GFLOPS")
        ax.set_xlabel("NNZ")

        ax.set_xscale('linear')  # 可切换为 log
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.set_ylim(0, 150)  # Y轴范围（可选）
        ax.legend()
        plt.tight_layout()

        fig_path = output_dir / f"gflops_line_k{k}.png"
        plt.savefig(fig_path, dpi=600)
        plt.close()

        print(f"The line chart was generated successfully! The file is stored in:{fig_path}")


if __name__ == "__main__":
    main()
