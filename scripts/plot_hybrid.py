import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import matplotlib

# ✅ 设置全局样式
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


def preprocess(df, k):
    df = df.sort_values(by="NNZ").reset_index(drop=True)
    df = df[(df["NNZ"] >= 10000) & (df["NNZ"] <= 2000000)]
    df = df.dropna(subset=["K", "BSMR"])
    df = df.drop_duplicates(subset=["file", "K"])
    df = df[df["K"] == k].copy().reset_index(drop=True)
    df = df.sort_values(by="NNZ").reset_index(drop=True)

    window_size = 5
    if len(df) < window_size:
        print(f"Skipping K={k} due to insufficient data points (<{window_size})")
        return None

    numeric_cols = ["NNZ", "BSMR", "BSMR_Only_Tensor_core", "BSMR_Only_CUDA_Core"]
    avg_df = df[numeric_cols].rolling(window=window_size).mean().dropna().reset_index(drop=True)
    return avg_df


def plot_hybrid_subplots(k_data_list, output_dir, output_name_suffix):
    num_plots = len(k_data_list)
    if num_plots == 0:
        print("没有有效的输入数据，无法绘图。请至少提供一个 CSV 文件。")
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))  # 每个图宽度为8
    if num_plots == 1:
        axes = [axes]  # 保证可迭代

    for ax, (k, avg_df) in zip(axes, k_data_list):
        x = avg_df["NNZ"].values
        ax.plot(x, avg_df["BSMR"], label="Hybrid", alpha=0.7)
        ax.plot(x, avg_df["BSMR_Only_Tensor_core"], label="Only Tensor Core", alpha=0.7)
        ax.plot(x, avg_df["BSMR_Only_CUDA_Core"], label="Only CUDA Core", alpha=0.7)

        ax.set_title(f"K = {k}")
        ax.set_xlabel("NNZ")
        ax.set_xscale('linear')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.set_ylabel("GFLOPS")

    plt.tight_layout()
    fig_path = output_dir / f"hybrid_{output_name_suffix}.png"
    plt.savefig(fig_path)
    plt.close()

    print(f"Saved figure to {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="测试结果可视化(可支持单个或多个K)")
    parser.add_argument('--k32', type=str, help="输入的 CSV 文件路径(K=32)")
    parser.add_argument('--k128', type=str, help="输入的 CSV 文件路径(K=128)")
    parser.add_argument('--outdir', type=str, default='.', help="图表输出目录(默认当前目录)")
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_data_list = []

    if args.k32:
        df_k32 = pd.read_csv(args.k32)
        avg_k32 = preprocess(df_k32, 32)
        if avg_k32 is not None:
            k_data_list.append((32, avg_k32))

    if args.k128:
        df_k128 = pd.read_csv(args.k128)
        avg_k128 = preprocess(df_k128, 128)
        if avg_k128 is not None:
            k_data_list.append((128, avg_k128))

    # 自动生成文件名后缀
    suffix_parts = [f'k{k}' for (k, _) in k_data_list]
    output_name_suffix = "_".join(suffix_parts)

    plot_hybrid_subplots(k_data_list, output_dir, output_name_suffix)


if __name__ == "__main__":
    main()
