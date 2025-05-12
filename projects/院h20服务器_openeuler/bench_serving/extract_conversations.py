#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path

SRC_FILE = Path("y:\\dataset.json")
DST_FILE = Path("y:\\dataset-1000.json")
NUM_CONV  = 1000                       # 需要抽取的 conversation 数量

def main() -> None:
    # 1. 读取源文件
    try:
        with SRC_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"❌ 找不到源文件: {SRC_FILE.resolve()}")
    except json.JSONDecodeError as e:
        sys.exit(f"❌ JSON 解析失败: {e}")

    # 2. 选取前 NUM_CONV 条 conversation
    if not isinstance(data, list):
        sys.exit("❌ 源文件顶层结构应为列表 (list)。")

    subset = data[:NUM_CONV]           # 如果不足 1000 条，会自动取全部

    # 3. 写出到目标文件
    with DST_FILE.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"✅ 已将 {len(subset)} 条 conversation 写入 {DST_FILE.resolve()}")

if __name__ == "__main__":
    main()
