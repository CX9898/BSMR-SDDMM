import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import matplotlib

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


def main():
    parser = argparse.ArgumentParser(description="测试结果可视化（折线图）")
    parser.add_argument('--file', type=str, required=True, help="输入的 CSV 结果文件路径")
    parser.add_argument('--outdir', type=str, default='.', help="图表输出目录（默认当前目录）")
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    df.rename(columns={
        "Matrix File": "file",
        "BSMR": "BSMR_gflops",
        "cuSDDMM": "cuSDDMM_gflops",
        "cusparse": "cuSparse_gflops",
        "RoDe": "RoDe_gflops",
        "TCGNN": "TCGNN_gflops"
    }, inplace=True)

    # 过滤和清洗数据
    df = df.sort_values(by="NNZ").reset_index(drop=True)
    df = df[(df["NNZ"] >= 10000) & (df["NNZ"] <= 2000000)]
    df = df.dropna(subset=["K", "BSMR_gflops"])
    df = df.drop_duplicates(subset=["file", "K"])

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_K = sorted(df["K"].unique())

    for k in unique_K:
        subset = df[df["K"] == k].copy().reset_index(drop=True)
        subset = subset.sort_values(by="NNZ").reset_index(drop=True)

        window_size = 5
        if len(subset) < window_size:
            print(f"Skipping K={k} due to insufficient data points (<{window_size})")
            continue

        numeric_cols = ["NNZ", "BSMR_gflops", "cuSDDMM_gflops", "cuSparse_gflops",
                         "RoDe_gflops", "TCGNN_gflops"]

        # 滑动平均只对数值列
        avg_subset = subset[numeric_cols].rolling(window=window_size).mean().dropna().reset_index(drop=True)

        x = avg_subset["NNZ"].values

        fig, ax = plt.subplots()
        ax.plot(x, avg_subset["BSMR_gflops"], label="BSMR", alpha=0.7)
        ax.plot(x, avg_subset["cuSDDMM_gflops"], label="cuSDDMM", alpha=0.7)
        ax.plot(x, avg_subset["cuSparse_gflops"], label="cuSparse", alpha=0.7)
        ax.plot(x, avg_subset["RoDe_gflops"], label="RoDe", alpha=0.7)
        ax.plot(x, avg_subset["TCGNN_gflops"], label="TCGNN", alpha=0.7)

        ax.set_title(f"K = {k}")
        ax.set_ylabel("GFLOPS")
        ax.set_xlabel("NNZ")
        ax.set_xscale('linear')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.set_ylim(bottom=0)
        ax.legend()
        plt.tight_layout()

        fig_path = output_dir / f"gflops_line_k{k}.png"
        plt.savefig(fig_path)
        plt.close()

        print(f"The line chart was generated successfully! The file is stored in: {fig_path}")


if __name__ == "__main__":
    main()
