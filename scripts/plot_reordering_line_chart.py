import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
import matplotlib
import matplotlib.ticker as ticker
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


def parse_dense_block_table(file_path):
    data = []
    parsing = False
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if re.match(r'\|\s*NNZ\s*\|\s*bsmr_numDenseBlock\s*\|\s*bsa_numDenseBlock\s*\|', line):
                parsing = True
                continue
            if parsing and line.strip().startswith("|") and not line.strip().startswith("|-"):
                parts = [x.strip() for x in line.strip().split("|")[1:-1]]
                if len(parts) == 3 and parts[0].isdigit():
                    try:
                        row = {
                            "NNZ": int(parts[0]),
                            "bsmr_numDenseBlock": int(parts[1]),
                            "bsa_numDenseBlock": int(parts[2])
                        }
                        data.append(row)
                    except ValueError:
                        continue
    return pd.DataFrame(data)


def plot_dense_blocks(ax, df):
    df = df.sort_values("NNZ").reset_index(drop=True)

    # 滑动平均（窗口大小=5）
    df["bsmr_smooth"] = df["bsmr_numDenseBlock"].rolling(window=5, min_periods=1).mean()
    df["bsa_smooth"] = df["bsa_numDenseBlock"].rolling(window=5, min_periods=1).mean()

    # 可选：子采样
    sampled = df.iloc[::3]

    # ax.plot(sampled["NNZ"], sampled["bsmr_smooth"], marker='o', color='tab:blue', markersize=4,
    #         label="bsmr_numDenseBlock", linewidth=2, alpha=0.9)
    # ax.plot(sampled["NNZ"], sampled["bsa_smooth"], marker='s', color='tab:orange', markersize=4,
    #         label="bsa_numDenseBlock", linewidth=2, alpha=0.9)

    ax.plot(sampled["NNZ"], sampled["bsmr_smooth"],
            linestyle='-', marker='o', color='#1f77b4', markersize=5,
            label="BSMR", linewidth=2)
    ax.plot(sampled["NNZ"], sampled["bsa_smooth"],
            linestyle='--', marker='s', color='#999999', markersize=5,
            label="BSA", linewidth=2)

    ax.set_xlabel("NNZ")
    ax.set_ylabel("Number of Dense Blocks")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.set_xscale('log')

    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc="upper left")


def main():
    parser = argparse.ArgumentParser(description="DenseBlock 折线图可视化（增强版）")
    parser.add_argument('-file', type=str, required=True, help="输入的 Markdown 文件路径")
    parser.add_argument('-outdir', type=str, default='.', help="图表输出目录")
    args = parser.parse_args()

    df = parse_dense_block_table(args.file)
    if df.empty:
        print("未找到表格数据，请确认 Markdown 格式正确。")
        return


    df = df.drop_duplicates().sort_values("NNZ").reset_index(drop=True)

    nnz_min = 1e5
    nnz_max = 1e6
    df = df[(df["NNZ"] >= nnz_min) & (df["NNZ"] <= nnz_max)]

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    plot_dense_blocks(ax, df)

    plt.tight_layout()
    plt.savefig(output_dir / "dense_block_line_chart.png", dpi=600)
    plt.savefig(output_dir / "dense_block_line_chart.pdf", format="pdf", bbox_inches="tight")
    print(f"图表已保存至: {output_dir}")


if __name__ == "__main__":
    main()
