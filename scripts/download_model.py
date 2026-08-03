"""通过 ModelScope 下载 Qwen2.5-3B-Instruct 到本地目录。"""
import os
import sys
from pathlib import Path

# 把项目根目录加入路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from modelscope import snapshot_download

MODEL_ID = "qwen/Qwen2.5-3B-Instruct"
LOCAL_DIR = project_root / "models" / "qwen2.5-3b-instruct"

def main():
    print(f"正在从 ModelScope 下载 {MODEL_ID}")
    print(f"目标目录: {LOCAL_DIR}")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, cache_dir=str(LOCAL_DIR))
    print("下载完成")

if __name__ == "__main__":
    main()

