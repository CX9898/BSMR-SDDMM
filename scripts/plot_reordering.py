import matplotlib.pyplot as plt
import argparse
import re
from collections import defaultdict


def parse_data(filepath):
    pattern = re.compile(
        r"Alpha: ([\d.]+), Delta: ([\d.]+), "
        r"BSMR num dense blocks: (\d+), BSA num dense blocks: (\d+), original num dense blocks: (\d+), "
        r"BSMR average density: ([\d.]+), BSA average density: ([\d.]+), original average density: ([\d.]+)"
    )

    data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                alpha = float(match.group(1))
                delta = float(match.group(2))
                bsmr_num = int(match.group(3))
                bsa_num = int(match.group(4))
                orig_num = int(match.group(5))
                bsmr_density = float(match.group(6))
                bsa_density = float(match.group(7))
                orig_density = float(match.group(8))

                data[alpha].append({
                    'delta': delta,
                    'bsmr_num': bsmr_num,
                    'bsa_num': bsa_num,
                    'orig_num': orig_num,
                    'bsmr_density': bsmr_density,
                    'bsa_density': bsa_density,
                    'orig_density': orig_density,
                })
    return data


def plot_data(data, output_file):
    alphas = sorted(data.keys())
    num_subplots = len(alphas)

    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5), constrained_layout=True)

    if num_subplots == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []
    added_labels = set()

    for idx, (ax, alpha) in enumerate(zip(axes, alphas)):
        group = sorted(data[alpha], key=lambda x: x['delta'])
        deltas = [d['delta'] for d in group]

        bsmr_nums = [d['bsmr_num'] for d in group]
        bsa_nums = [d['bsa_num'] for d in group]
        orig_nums = [d['orig_num'] for d in group]

        bsmr_density = [d['bsmr_density'] for d in group]
        bsa_density = [d['bsa_density'] for d in group]
        orig_density = [d['orig_density'] for d in group]

        bar_width = 0.2
        x = range(len(deltas))

        h1 = ax.bar([i - bar_width for i in x], bsmr_nums, width=bar_width, label='BSMR Average Number of Dense Blocks',
                    color='skyblue')
        h2 = ax.bar(x, bsa_nums, width=bar_width, label='BSA Average Number of Dense Blocks', color='lightgreen')
        h3 = ax.bar([i + bar_width for i in x], orig_nums, width=bar_width,
                    label='Original Average Number of Dense Blocks', color='lightgray')

        ax2 = ax.twinx()
        l1, = ax2.plot(x, bsmr_density, marker='o', label='BSMR Average Density of Dense Blocks', color='blue')
        l2, = ax2.plot(x, bsa_density, marker='s', label='BSA Average Density of Dense Blocks', color='green')
        l3, = ax2.plot(x, orig_density, marker='^', label='Original Average Density of Dense Blocks', color='gray')

        ax.set_title(f'Alpha = {alpha}')

        # 左边柱状图 Y 轴标签
        if idx == 0:
            ax.set_ylabel('Average Number of Dense Blocks')
        else:
            ax.set_ylabel('')  # 清除标签（但不隐藏刻度）

        # 右边折线图 Y 轴标签
        if idx == len(alphas) - 1:
            ax2.set_ylabel('Average Density of Dense Blocks')
        else:
            ax2.set_ylabel('')  # 清除标签（但保留刻度）

        ax.set_xlabel('Delta')
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in deltas])

        for handle, label in zip(
                [h1[0], h2[0], h3[0], l1, l2, l3],
                ['BSMR Average Number of Dense Blocks', 'BSA Average Number of Dense Blocks',
                 'Original Average Number of Dense Blocks', 'BSMR Average Density of Dense Blocks',
                 'BSA Average Density of Dense Blocks',
                 'Original Average Density of Dense Blocks']
        ):
            if label not in added_labels:
                legend_handles.append(handle)
                legend_labels.append(label)
                added_labels.add(label)

    # 顶部居中图例
    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        ncol=6,  # 设置一行显示几个图例项
        fontsize='medium',
        bbox_transform=fig.transFigure,
        bbox_to_anchor=(0.5, 1.10)  # 微调高度让 legend 更居中
    )

    # 不再调用 tight_layout
    fig.savefig(output_file, bbox_inches='tight')
    print(f"图已保存为 {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='输入数据文件路径')
    parser.add_argument('--output', default='evaluateReordering.png', help='输出图像文件名')
    args = parser.parse_args()

    data = parse_data(args.file)
    plot_data(data, args.output)
