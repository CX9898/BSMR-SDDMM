import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import matplotlib

matplotlib.rcParams['figure.figsize'] = (14, 7)


def parse_markdown_data(file_path):
    data = []
    current_file = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            file_match = re.search(r'file\s*:\s*.*/(.+\.mtx)', line)
            if file_match:
                current_file = file_match.group(1).strip()
            elif line.strip().startswith("|") and not line.strip().startswith("| M ") and current_file:
                parts = [x.strip().replace('%','') for x in line.strip().split("|")[1:-1]]
                if len(parts) >= 10:
                    try:
                        row = {
                            "file": current_file,
                            "NNZ": int(parts[2]),
                            "K": int(parts[4]),
                            "zcx_gflops": float(parts[5]) if parts[5] else None,
                            "cuSDDMM_gflops": float(parts[6]) if parts[6] else None,
                            "cuSparse_gflops": float(parts[7]) if parts[7] else None,
                            "RoDe_gflops": float(parts[8]) if parts[8] else None,
                            "ASpT_gflops": float(parts[9]) if parts[9] else None
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
    df = df.sort_values(by="NNZ").reset_index(drop=True)
    df = df.dropna(subset=["K", "zcx_gflops"])
    df = df.drop_duplicates(subset=["file", "K"])

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_K = sorted(df["K"].unique())

    for k in unique_K:
        subset = df[df["K"] == k].copy().reset_index(drop=True)
        subset = subset.fillna(0).infer_objects(copy=False)
        subset = subset.iloc[::3].copy().reset_index(drop=True)  # 子采样
        x_labels = subset["NNZ"].astype(str).tolist()
        x = np.arange(len(x_labels))
        width = 0.15

        fig, ax = plt.subplots()
        ax.bar(x - 2*width, subset["cuSDDMM_gflops"], width, label="cuSDDMM")
        ax.bar(x - width, subset["cuSparse_gflops"], width, label="cuSparse")
        ax.bar(x, subset["zcx_gflops"], width, label="zcx")
        ax.bar(x + width, subset["RoDe_gflops"], width, label="RoDe")
        ax.bar(x + 2*width, subset["ASpT_gflops"], width, label="ASpT")

        ax.set_title(f"GFLOPS Comparison at K={k}")
        ax.set_ylabel("GFLOPS")
        ax.set_xlabel("NNZ")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=60, ha='right', fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))  # ← 智能选择最多 12 个标签
        ax.legend()
        plt.tight_layout()

        fig_path = output_dir / f"gflops_bar_k{k}.png"
        plt.savefig(fig_path, dpi=600)
        plt.close()


if __name__ == "__main__":
    main()
