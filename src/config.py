"""实验配置常量。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 本地模型路径（ModelScope 下载后的实际快照目录）
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen2.5-3b-instruct"
    / "models"
    / "qwen--Qwen2.5-3B-Instruct"
    / "snapshots"
    / "master"
)

# 数据目录
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 实验输出目录
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"

# 默认模型生成参数
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMPERATURE = 0.7

# 长文档切分默认参数
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

