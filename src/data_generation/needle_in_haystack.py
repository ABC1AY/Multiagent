"""生成 needle-in-a-haystack 合成长文档数据集。"""
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

# 支持直接运行该文件
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import DATA_PROCESSED_DIR

# 一些用于填充文档的通用主题段落模板
FILLER_PARAGRAPHS = [
    "人工智能（AI）是计算机科学的一个分支，致力于创造能够执行通常需要人类智能的任务的系统。",
    "机器学习是 AI 的核心方法之一，通过从数据中学习模式来做出预测或决策。",
    "深度学习使用多层神经网络，能够从大量数据中提取复杂的特征表示。",
    "自然语言处理让计算机能够理解、解释和生成人类语言，包括文本和语音。",
    "强化学习通过与环境交互并接收奖励信号，使智能体学会在特定任务中做出最优决策。",
    "计算机视觉使机器能够从图像和视频中提取信息，广泛应用于人脸识别、自动驾驶等领域。",
    "大语言模型通过在海量文本上训练，获得了强大的语言理解和生成能力。",
    "数据预处理是机器学习流程中的重要步骤，包括清洗、归一化和特征工程。",
    "模型评估需要使用独立的测试集，并选择合适的指标来衡量性能。",
    "过拟合是指模型在训练数据上表现很好，但在未见过的数据上表现较差的现象。",
    "迁移学习允许模型将在一个任务上学到的知识应用到另一个相关任务上。",
    "注意力机制帮助模型在处理序列时聚焦于最相关的部分，显著提升了翻译和摘要效果。",
    "生成对抗网络由生成器和判别器组成，两者相互博弈以生成逼真的数据。",
    "知识图谱以图的形式组织实体和关系，为推理和问答提供了结构化的知识基础。",
    "多模态学习同时处理文本、图像、音频等多种数据类型，是实现更通用 AI 的重要方向。",
]

# Needle 事实模板
NEEDLE_TEMPLATES = [
    {
        "needle": "{name}的生日是 {birthday}。",
        "question": "{name}的生日是什么时候？",
        "answer": "{birthday}",
    },
    {
        "needle": "{name}毕业于 {university}。",
        "question": "{name}毕业于哪所大学？",
        "answer": "{university}",
    },
    {
        "needle": "{name}的邮箱地址是 {email}。",
        "question": "{name}的邮箱地址是什么？",
        "answer": "{email}",
    },
    {
        "needle": "{name}最喜欢的颜色是 {color}。",
        "question": "{name}最喜欢的颜色是什么？",
        "answer": "{color}",
    },
]

POSITION_NAMES = ["front", "middle", "back"]


def generate_random_fact():
    """生成一条随机的 needle 事实。"""
    names = ["张三", "李四", "王五", "赵六", "陈七", "刘八", "孙九", "周十"]
    universities = ["清华大学", "北京大学", "浙江大学", "上海交通大学", "复旦大学", "中国科学技术大学"]
    colors = ["红色", "蓝色", "绿色", "紫色", "黄色", "黑色", "白色"]
    template = random.choice(NEEDLE_TEMPLATES)
    fact = {
        "name": random.choice(names),
        "birthday": f"{random.randint(1980, 2000)}年{random.randint(1, 12)}月{random.randint(1, 28)}日",
        "university": random.choice(universities),
        "email": f"{random.choice(names)}_{random.randint(100,999)}@example.com",
        "color": random.choice(colors),
    }
    return {
        "needle": template["needle"].format(**fact),
        "question": template["question"].format(**fact),
        "answer": template["answer"].format(**fact),
    }


def generate_document(target_length: int, needle: str, position: str) -> str:
    """生成一篇指定长度的文档，并在指定位置插入 needle。

    Args:
        target_length: 目标 token 数（近似，按中文字符估算）。
        needle: 要插入的事实句子。
        position: "front", "middle", "back" 之一。
    """
    # 估算：每个 filler 段落约 40-60 个中文字符
    chars_per_para = 50
    num_paras = max(3, target_length // chars_per_para)

    paragraphs = [random.choice(FILLER_PARAGRAPHS) for _ in range(num_paras)]

    if position == "front":
        insert_idx = len(paragraphs) // 6
    elif position == "back":
        insert_idx = 5 * len(paragraphs) // 6
    else:  # middle
        insert_idx = len(paragraphs) // 2

    paragraphs.insert(insert_idx, needle)
    return "\n\n".join(paragraphs)


def generate_dataset(
    num_samples: int = 100,
    lengths: list[int] | None = None,
    positions: list[str] | None = None,
    output_path: Path | None = None,
) -> list[dict]:
    """生成 needle-in-haystack 数据集并保存为 JSONL。"""
    lengths = lengths or [4096, 8192, 16384, 32768]
    positions = positions or POSITION_NAMES
    output_path = output_path or (DATA_PROCESSED_DIR / "needle_in_haystack.jsonl")
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    samples = []
    for i in tqdm(range(num_samples), desc="Generating needle-in-haystack"):
        fact = generate_random_fact()
        length = random.choice(lengths)
        position = random.choice(positions)
        document = generate_document(length, fact["needle"], position)
        samples.append(
            {
                "id": i,
                "document": document,
                "question": fact["question"],
                "answer": fact["answer"],
                "needle": fact["needle"],
                "target_length": length,
                "position": position,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Generated {len(samples)} samples -> {output_path}")
    return samples


if __name__ == "__main__":
    generate_dataset(num_samples=100)

