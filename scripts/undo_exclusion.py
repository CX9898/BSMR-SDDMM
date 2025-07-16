import os
import shutil
import sys

def parse_log(log_file):
    entries = []
    current = {}
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("original:"):
                current["original"] = line[len("original:"):].strip()
            elif line.startswith("moved_to:"):
                current["moved_to"] = line[len("moved_to:"):].strip()
            elif line == "---":
                if "original" in current and "moved_to" in current:
                    entries.append(current)
                current = {}
    return entries

def undo_moves(log_file):
    if not os.path.isfile(log_file):
        print(f"日志文件不存在: {log_file}")
        return

    entries = parse_log(log_file)
    print(f"共检测到 {len(entries)} 条移动记录，开始撤回...")

    for entry in entries:
        src = entry["moved_to"]
        dst = entry["original"]

        if not os.path.isfile(src):
            print(f"[跳过] 文件不存在: {src}")
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        print(f"[还原] {src} -> {dst}")

    print("撤回操作完成。")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python undo_exclusion.py /path/to/excluded_dataset/dataset_name/excluded_files.log")
        sys.exit(1)

    log_path = os.path.abspath(sys.argv[1])
    undo_moves(log_path)
