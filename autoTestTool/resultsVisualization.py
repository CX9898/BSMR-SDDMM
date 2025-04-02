import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
import matplotlib.ticker as ticker
import matplotlib

matplotlib.rcParams['figure.figsize'] = (12, 6)


def parse_markdown_data(file_path):
    data = []
    current_file = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            file_match = re.search(r'M\s*:\s*\d+,\s*N\s*:\s*\d+,\s*sparsity\s*:\s*[\d.]+%,\s*file\s*:\s*.*/(.+)', line)
            if file_match:
                current_file = file_match.group(1).strip()
            elif line.strip().startswith("|") and not line.strip().startswith("| M") and current_file:
                parts = [x.strip() for x in line.strip().split("|")[1:-1]]
                if len(parts) >= 7:
                    try:
                        row = {
                            "file": current_file,
                            "K": int(parts[3]),
                            "isratnisa_gflops": float(parts[4]) if parts[4] else None,
                            "cuSparse_gflops": float(parts[5]) if parts[5] else None,
                            "zcx_gflops": float(parts[6]) if parts[6] else None
                        }
                        data.append(row)
                    except ValueError:
                        continue
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="测试结果可视化")
    parser.add_argument('-file', type=str, required=True, help="输入的 Markdown 结果文件路径")
    parser.add_argument('-outdir', type=str, default='.', help="图表输出目录（默认当前目录）")
    args = parser.parse_args()

    df = parse_markdown_data(args.file)
    df = df.dropna(subset=["K", "zcx_gflops"])
    df = df.drop_duplicates(subset=["file", "K"])  # 去重，防止重复条目

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_K = sorted(df["K"].unique())

    for k in unique_K:
        subset = df[df["K"] == k].copy().reset_index(drop=True)
        x_labels = subset["file"].tolist()
        x = np.arange(len(x_labels))
        width = 0.25

        fig, ax = plt.subplots()
        ax.bar(x - width, subset["isratnisa_gflops"], width, label="isratnisa")
        ax.bar(x, subset["cuSparse_gflops"], width, label="cuSparse")
        ax.bar(x + width, subset["zcx_gflops"], width, label="zcx")

        ax.set_title(f"GFLOPS Comparison at K={k}")
        ax.set_ylabel("GFLOPS")
        ax.set_xlabel("Dataset File")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend()
        plt.tight_layout()

        # Save figure instead of showing it
        fig_path = output_dir / f"gflops_k{k}.png"
        plt.savefig(fig_path)
        plt.close()


if __name__ == "__main__":
    main()
