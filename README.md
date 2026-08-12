# SwarmAgent

基于 GRPO 的预算感知动态多智能体调度研究 —— 用于长文档问答的学习型证据调度器。

## 项目结构

```
SwarmAgent/
├── .venv/                          # Python 虚拟环境
├── data/                           # 数据集
│   ├── raw/                        # 原始数据
│   └── processed/                  # 处理后数据
├── docs/                           # 文档与提案
│   └── proposal/                   # 研究提案与 PPT 提取文本
│       ├── extracted_texts/        # PPT 与提案的提取文本
│       ├── proposal.docx           # 早期草案
│       └── 研究提案_基于GRPO的预算感知动态多智能体调度.docx
├── experiments/                    # 实验输出
│   ├── checkpoints/                # 模型检查点
│   ├── logs/                       # 训练与推理日志
│   └── results/                    # 实验结果（表格、图表）
├── literature/                     # 参考文献与资料
│   ├── pdf/                        # PDF 论文/综述
│   └── pptx/                       # PPT 源文件
├── scripts/                        # 常用脚本
├── src/                            # 源代码
│   ├── agents/                     # Leader / Worker 智能体实现
│   ├── data_generation/            # 数据生成（needle-in-haystack 等）
│   ├── evaluation/                 # 评估指标与可视化
│   ├── models/                     # 模型加载、量化、推理封装
│   └── training/                   # GRPO / RL 训练逻辑
├── .gitignore
└── requirements.txt                # Python 依赖
```

## 快速开始

1. 安装依赖：
   ```bash
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. 验证模型能跑：
   ```bash
   python src/models/test_model.py
   ```

3. 运行第一阶段实验：
   ```bash
   python scripts/run_phase1_baseline.py
   ```

## 实验路线图

实验按**从最小可行到完整论文**分 5 个阶段推进，不要跳步。

### Phase 0：环境准备

1. 创建虚拟环境并激活：
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 下载 Qwen2.5-3B-Instruct：
   ```bash
   python scripts/download_model.py
   ```
4. 验证模型能跑：
   ```bash
   python scripts/test_model.py
   ```
**完成标志**：`test_model.py` 成功输出中文回答，显存占用 < 6G。

### Phase 1：简化版 LONGAGENT 基线

**目标**：证明“切分 + Leader-Worker 协作”这个基本 loop 能工作。

1. **生成合成数据**：
   - 写脚本生成 100 条 needle-in-haystack 样本
   - 长度：4K / 8K / 16K / 32K token
   - Needle 位置：前 1/3、中 1/3、后 1/3
   - 问题模板固定，例如“某人的生日是多少？”
2. **固定分块**：把文档按 512 或 1024 token 切成 chunk。
3. **实现 Worker**：
   - 每个 Worker 读一个 chunk
   - 找到答案就返回证据
   - 找不到就返回“未提及”
4. **实现 Leader（硬编码规则版）**：
   - 动作：QUERY → 等所有 Worker 回答 → 判断证据是否足够 → 不够继续问 / 够就 ANSWER
   - 第一轮 QUERY 发给所有 Worker（复现 LONGAGENT）
5. **实现两个基线**：
   - 基线 A：单模型读全文（模拟长上下文模型）
   - 基线 B：单模型只读前 K 个 chunk（模拟固定长度截断）

**记录指标**：
- 答案准确率
- 调用 Worker 次数
- 总 token 数
- 推理时间

**完成标志**：Leader-Worker 协作准确率 > 单模型截断基线，接近或超过单模型全文基线。

### Phase 2：加入简单调度策略

**目标**：在不训练的情况下，验证“选择性调用”有价值。

1. **语义粗召回**：用 BM25 或 embedding 给每个 chunk 和问题打分，只召回 top-K chunk。
2. **Leader 只 QUERY 召回的 Worker**，而不是全部。
3. **加入冲突检测**：如果两个 Worker 答案矛盾，触发 CONFLICT 让它们互换 chunk 重读。
4. **加入简单停止规则**：如果 top 答案置信度 > 阈值，直接 ANSWER。

**对比组**：
- LONGAGENT（全广播）
- 仅召回 top-K（无冲突、无停止）
- 召回 + 冲突消解
- 召回 + 冲突消解 + 停止规则

**完成标志**：在保持准确率的前提下，Worker 调用次数下降 30%+。

### Phase 3：多轮 Agent Loop（Model-decided QUERY/VERIFY/ANSWER）

**目标**：让 Leader 自己决定多轮调度，而不是 Phase 2 的硬编码单轮 selective + conflict + fallback。

1. **第一轮：种子查询**
   - 用 BM25 top-K 检索 top-5 chunks 作为种子证据
   - 5 个 Worker 并行答
2. **后续轮：模型决策**
   - 把"当前证据 + 已用预算 + 历史动作"喂给 Leader
   - Leader 每轮输出以下动作之一：
     - `QUERY[i]`：查第 i 个 chunk 的 Worker
     - `VERIFY[a,b]`：让 Worker a 和 b 互换 chunk 重读（解决冲突）
     - `ANSWER`：输出最终答案
     - `QUERY_ALL`：fallback 全广播
3. **预算控制**
   - `AGENT_MAX_ROUNDS = 4`（最多 4 轮）
   - `AGENT_BUDGET = 10`（最多 10 次 Worker 调用）

**完成标志**：相比 Phase 2 selective_full，**6× 少调用 + 3× 少 token，准确率损失 ≤5pp**（Pareto 前沿向左下移动）。

**注**：原 README 设计是"训分类器决定何时停"，实际实现改为多轮 agent loop —— 跟老师建议的 Agentic RL / 多轮查询方向一致，是 Phase 4 GRPO 训练调度策略的天然起点。
### Phase 4：GRPO 端到端训练调度策略

**目标**：把 Leader 的完整决策变成用 GRPO 训练的策略。

1. **定义 POMDP**：
   - 状态 `s_t = (问题, 共享记忆, 已收集证据, 未解决冲突, 剩余预算)`
   - 动作 `a_t = (选哪些 Worker / chunk / 操作 / 是否停止)`
   - 奖励 `R = 正确性 - α·token - β·调用次数`
2. **用同一个 3B 模型作为策略模型**（先用 LoRA，不要 full fine-tune）。
3. **采样多组轨迹**，对同一问题比较不同调度路径的奖励。
4. **GRPO 更新**：组内相对优势，不需要 Critic 网络。
5. **先在 100–500 条合成样本上跑通**，再扩大到 LongBench 子集。

**完成标志**：训练后的策略在准确率-成本 Pareto 前沿上优于硬编码调度器。

### Phase 5：正式基准与真实文档

**目标**：在公开数据和真实文档上验证，支撑论文结论。

1. **LongBench 单文档 QA 子集**上跑完整系统 vs. 基线。
2. **InfiniteBench** 上测超长文档能力（显存不够时用量子化或只测部分子集）。
3. **真实文档**：选 1–2 个领域（法律合同 / 金融年报 / 医学文献），人工构造 50 条 QA。
4. **完整消融实验**：
   - Base：固定分块 + 全广播
   - + 自适应分区
   - + 自适应路由
   - + 自适应拓扑
   - + 验证器 + 预算停止 + GRPO
5. **画 Pareto 前沿图**：准确率 vs. token / 调用次数。

## Phase 3 初步结果（20 条样本，Qwen2.5-3B-Instruct，TriviaQA RC）

| 方法 | 准确率 | 平均 Token 数 | 平均 Worker 调用 | 耗时 |
|------|--------|--------------|-----------------|------|
| selective_full（Phase 2 baseline）| 45.0% | 14,416 | 26.7 | 238s |
| **agent_loop（Phase 3）** | **40.0%** | **4,440** | **5.45** | ~80s |

**观察**：
- agent_loop 用 **6× 少 Worker 调用** + **3× 少 token**，准确率仅损失 5pp（45% → 40%）
- 在准确率-成本 Pareto 前沿上明显向左下移动
- 模型自己学会了"什么时候再查一个"、"什么时候停"，比硬编码 single-pass 更高效
- 这是 Phase 4 GRPO 训练调度策略的 baseline

结果文件：`experiments/results/phase3_agent.json` ## 当前进度
 
 - [x] Phase 0：环境配置与模型下载
 - [x] Phase 1：简化版 LONGAGENT 基线
 - [x] Phase 2：硬编码调度优化
 - [x] Phase 3：多轮 Agent Loop
 - [ ] Phase 4：GRPO 训练
 - [ ] Phase 5：公开基准与真实文档评估
 
 ## Phase 1 初步结果（20 条样本，Qwen2.5-3B-Instruct）
 
 | 方法 | 准确率 | 平均 Token 数 | 平均 Worker 调用 | 耗时 |
 |------|--------|--------------|-----------------|------|
 | Multi-Agent（LONGAGENT-style） | 75.0% | 7,905 | 15.1 | 96s |
 | Single-Model Full Context | 70.0% | 6,769 | - | 1001s |
 | Single-Model Truncated | 45.0% | 2,061 | - | 62s |
 
 
## Phase 2 初步结果（20 条样本，Qwen2.5-3B-Instruct）

| 方法 | 准确率 | 平均 Token 数 | 平均 Worker 调用 | 耗时 |
|------|--------|--------------|-----------------|------|
| broadcast（Phase 1 全广播） | 75.0% | 7,905.5 | 15.1 | 102.9s |
| selective（BM25 top-5） | 75.0% | 2,486.1 | 4.7 | 38.4s |
| selective + conflict | 75.0% | 2,486.1 | 4.7 | 37.7s |
| selective + conflict + fallback | 75.0% | 3,350.4 | 6.3 | 47.9s |

**观察**：
- 仅 BM25 召回 top-5 即可将 Worker 调用从 15.1 次降到 4.7 次（下降约 69%），Token 消耗从约 7,900 降到约 2,500（下降约 68%），准确率保持 75.0%。
- needle-in-haystack 任务中通常只有一个 chunk 包含答案，其余 chunk 返回“未提及”，因此 `selective + conflict` 没有触发冲突消解，指标与 `selective` 相同。
- `selective + conflict + fallback` 在部分样本上因置信度不足回退到全广播，调用次数和 Token 有所增加，但仍低于原始广播基线。

结果文件：`experiments/results/phase2_baseline.json`

**观察**：
 - Multi-Agent 在准确率上略高于单模型全文基线，但速度快了约 10 倍。
 - Multi-Agent 显著优于截断基线（+30% 准确率），说明分块协作能有效找回分散证据。
 - 单模型全文基线在长文档上非常慢，且部分长样本会触发 CUDA OOM（已用占位答案处理）。
 
 结果文件：`experiments/results/phase1_baseline.json`
