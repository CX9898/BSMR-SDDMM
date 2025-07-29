import argparse
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import re
import os
from pandas import DataFrame
from collections import defaultdict
import warnings

PLOT_ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
BAR_WIDTH = 0.15


def parse_text_to_dataframe(filepath):
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(
                r'Alpha: ([\d.]+), m in \[(\d+), (\d+)\), Num Results: (\d+), Avg Clusters: ([\d.]+), Avg Row Reordering Time: ([\d.]+) ms',
                line.strip())
            if match:
                alpha = float(match.group(1))
                m_start = int(match.group(2))
                m_end = int(match.group(3))
                m_mid = (m_start + m_end) // 2
                cluster_cnt = float(match.group(5))
                reorder_time = float(match.group(6))
                records.append({
                    'rows': m_start,
                    'alpha': alpha,
                    'cluster_cnt': cluster_cnt,
                    'avg_reordering_time': reorder_time
                })
    return pd.DataFrame(records)


def configured_plt(df, x_ticks):
    plt.rcParams.update({'font.size': 15})
    fig, time_ax = plt.subplots(figsize=(12, 6))
    clstr_ax = time_ax.twinx()

    time_ax.set_yscale('log', base=2)
    clstr_ax.set_yscale('log', base=2)
    time_ax.set_xlabel("Number of Rows,Columns", fontsize=19)
    time_ax.set_ylabel('Average Elapsed Time (ms)', fontsize=19)
    clstr_ax.set_ylabel('Average Number of Clusters', fontsize=19)

    clstr_ax.set_ylim([2, df['cluster_cnt'].max() * 2])
    time_ax.set_ylim([1, df['avg_reordering_time'].max() * 4])
    plt.xticks([i for i in range(len(x_ticks))], x_ticks)
    return fig, time_ax, clstr_ax


# 生成 5 个从浅到深的蓝色
cmap = matplotlib.colormaps['Blues']  # 替代 cm.get_cmap('Blues')
bar_colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0.4, 0.9, len(PLOT_ALPHAS))]

# bar_colors = [
#     '#4e79a7',  # 深蓝
#     '#f28e2b',  # 橘橙
#     '#76b7b2',  # 青绿色
#     '#e15759',  # 红色
#     '#b07aa1',  # 紫色
# ]
line_colors = [
    '#1b9e77',  # 绿色
    '#d95f02',  # 橙红
    '#7570b3',  # 紫蓝
    '#e7298a',  # 粉红
    '#66a61e',  # 草绿
    '#e6ab02',  # 黄褐
    '#a6761d',  # 棕色
    '#666666',  # 深灰
    '#1f78b4',  # 蓝色
    '#fb9a99',  # 淡粉（对比红色柱子）
]
hatch_styles = ['//', '\\\\', '.', 'x', 'o']


def plot_reordering_bar(plot_df, time_ax, row_vals):
    count = 0
    color = iter(cm.viridis_r(np.linspace(0, 1, len(PLOT_ALPHAS))))
    for idx, alpha in enumerate(PLOT_ALPHAS):
        tmp = plot_df[plot_df.alpha == alpha].sort_values(by="rows")
        elapsed = np.array(tmp.avg_reordering_time)
        x_offset = np.ones(len(elapsed)) * (-1 * len(PLOT_ALPHAS) / 2 + count) * BAR_WIDTH + np.arange(len(elapsed))
        if len(PLOT_ALPHAS) % 2 == 1:
            x_offset += BAR_WIDTH / 2

        c = bar_colors[idx]
        hatch = hatch_styles[idx % len(hatch_styles)]
        time_ax.bar(x_offset, elapsed, label=r"$\alpha$ = {}".format(alpha), width=BAR_WIDTH, color=c, zorder=-1,
                    edgecolor='black')

        if idx == 0:
            x_start, y_start = x_offset[0], elapsed[0]
        elif idx == len(PLOT_ALPHAS) - 1:
            x_end, y_end = x_offset[0], elapsed[0]
            plt.text(x_end - 0.3, y_end - 700, fr"{(y_end / y_start): .1f}$\times$",
                     color="red", fontsize=14, weight="bold")
            time_ax.annotate("", (x_end + 0.1, y_end), (x_start - 0.1, y_start),
                             arrowprops=dict(arrowstyle="<->", linewidth=2, color="red"))
        count += 1


def plot_numcluster_line(plot_df, clstr_ax, row_vals):
    color = iter(cm.Dark2(np.linspace(0, 1, len(row_vals))))
    for idx, row_size in enumerate(row_vals):
        tmp = plot_df[plot_df.rows == row_size].sort_values(by="alpha")
        tmp = tmp[tmp.alpha.isin(PLOT_ALPHAS)]
        cluster_arr = np.array(tmp.cluster_cnt)
        x_offset = idx + BAR_WIDTH * np.arange(len(cluster_arr)) - len(PLOT_ALPHAS) * BAR_WIDTH / 2
        if len(PLOT_ALPHAS) % 2 == 1:
            x_offset += BAR_WIDTH / 2

        # c = next(color)
        c = line_colors[idx]
        clstr_ax.plot(x_offset, cluster_arr, linestyle='--', marker='o', c=c, label=f"{row_size}", zorder=-1)

        if idx == 0:
            x_start, y_start = x_offset[0], cluster_arr[0]
            x_end, y_end = x_offset[-1], cluster_arr[-1]
            plt.text(x_end + 0.05, y_end - 140, fr"{(y_end / y_start): .0f}$\times$",
                     color="blue", fontsize=14, weight="bold")
            clstr_ax.annotate("", (x_end, y_end), (x_start, y_start),
                              arrowprops=dict(arrowstyle="<->", linewidth=2, color="blue"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='输入数据文件路径')
    parser.add_argument('--outdir', default='./', help='输出图像目录')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    SAVE_FIG_FNAME = os.path.join(args.outdir, 'evaluate_reordering_overhead.png')

    df = parse_text_to_dataframe(args.file)
    df = df[df.alpha.isin(PLOT_ALPHAS)]

    row_vals = sorted(df['rows'].unique())
    fig, t_ax, c_ax = configured_plt(df, row_vals)
    plot_reordering_bar(df, t_ax, row_vals)
    plot_numcluster_line(df, c_ax, row_vals)

    # 合并所有文本元素
    for f in t_ax.texts + c_ax.texts:
        fig.texts.append(f)
    t_ax.legend(labelspacing=0.16, handlelength=1.8, fontsize=13, loc='upper left')

    # 保存 PNG 高分辨率版本
    plt.savefig(SAVE_FIG_FNAME, bbox_inches='tight', dpi=300)

    # 保存 PDF 矢量图版本
    pdf_output_file = SAVE_FIG_FNAME.replace('.png', '.pdf')
    plt.savefig(pdf_output_file, bbox_inches='tight')

    print(f"Saved PNG to {SAVE_FIG_FNAME}")
    print(f"Saved PDF to {pdf_output_file}")
