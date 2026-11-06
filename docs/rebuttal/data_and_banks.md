# 数据、标签与 criterion bank

先复用保存的评测数据、2,000 条数学 criterion 和 2,545 条代码 criterion。完整运行需要两份 embedding cache 和 LiveCodeBench v1 的 400 题文件；这些大文件放在实验服务器的资产目录。新同学拿到仓库后，先完成下述资产检查，再连接模型服务。

## 1. 安装与检查资产

仓库已带 12 个评测文件和两个 bank。外部目录使用相同的 `datasets/`、`pools/` 布局。`MASRUBRIC_ASSET_SOURCE` 指向管理员提供的原资产目录；`MASRUBRIC_ASSETS_ROOT` 是本次实验使用的目录。

```bash
export MASRUBRIC_ASSET_SOURCE=/path/to/preserved/assets
export MASRUBRIC_ASSETS_ROOT="$PWD/paper/assets"
python scripts/rebuttal/prepare_assets.py install --mode symlink --strict
python scripts/rebuttal/prepare_assets.py inventory --strict
```

同一服务器用软链接即可。迁移到独立存储时改为 `--mode copy`。已有目标文件会被保留，安装器不会覆盖；盘点会继续检查目标文件。只补三个外部文件时可直接指定：

```bash
python scripts/rebuttal/prepare_assets.py install \
  --livecode-file /path/to/livecodebench_v1.jsonl \
  --math-cache /path/to/math-cache.jsonl \
  --code-cache /path/to/code-cache.jsonl --strict
```

`--report /path/to/new-assets-report.json` 保存机器可读盘点，报告文件也必须是新路径。检查包含记录数、题目 ID 重复、必要字段、criterion 名称、有限且非零的向量、统一维度和 bank/cache 的完整名称对应。`--embedding-dimension N` 可按实际 embedding 服务强制维度。cache 还须来自运行时使用的同一 embedding 模型及文本策略；改用新 embedding 模型时重新生成整个 cache。

### 全新clone，没有原服务器资产时

以下命令在仓库根目录执行，用于补齐三个外部文件。先部署与原bank一致的 `Qwen3-Embedding-8B` 服务，配置 `EMBEDDING_URL`、`EMBEDDING_MODEL`、`EMBEDDING_KEY`。缓存生成会调用该服务；不需要人工标签。

```bash
# 仅在LiveCodeBench尚未安装时下载。也可把main换成选定的上游revision。
python scripts/rebuttal/prepare_assets.py fetch-livecode \
  --assets-root paper/assets --source-revision main

python scripts/rebuttal/build_bank.py embed \
  --input-file paper/assets/pools/math/deduped-mixed_metrics_two_pool.json \
  --output-dir runs/bootstrap/math-embeddings --execute
python scripts/rebuttal/build_bank.py embed \
  --input-file paper/assets/pools/code/deduplicated_metrics_pool.json \
  --output-dir runs/bootstrap/code-embeddings --execute

python scripts/rebuttal/prepare_assets.py install \
  --assets-root "$PWD/paper/assets" \
  --math-cache "$PWD/runs/bootstrap/math-embeddings/embeddings.jsonl" \
  --code-cache "$PWD/runs/bootstrap/code-embeddings/embeddings.jsonl" \
  --embedding-dimension 4096 --strict
python scripts/rebuttal/prepare_assets.py inventory \
  --assets-root "$PWD/paper/assets" --embedding-dimension 4096 --strict
```

重建缓存与原存档缓存分开标记。每步输出目录必须是新的；已完成时直接使用原输出。LiveCodeBench下载会检查保存的400题身份和顺序，所用上游revision下的题面/测试与历史导出的内容可能有差异；复查历史表格时使用原存档。

## 2. 固定评测集合

以下路径均相对于 `paper/assets/` 或外部资产根目录。分母来自当前 `configs/paper_main.json`；新运行保留这些题目和顺序。

| Benchmark | 题数 | 文件 | 判题参考 |
| --- | ---: | --- | --- |
| GSM8K | 1319 | `datasets/gsm8k/gsm8k_id.jsonl` | `answer` |
| MATH-500 | 500 | `datasets/math500/test_ids.jsonl` | `answer` / `solution` |
| AQuA | 254 | `datasets/aqua/aqua_id.jsonl` | `correct`，选项题 |
| AMC23 | 40 | `datasets/amc23/amc23_data.json` | `answer` |
| OlympiadBench | 675 | `datasets/olympiad/test.jsonl` | `final_answer`，独立旧版 grader 环境 |
| OlymMATH Easy | 100 | `datasets/olymMATH/OlymMATH-EN-EASY.jsonl` | `answer`，math-verify |
| OlymMATH Hard | 100 | `datasets/olymMATH/OlymMATH-EN-HARD.jsonl` | `answer`，math-verify |
| AIME24 | 30 | `datasets/aime24/test.jsonl` | `answer` |
| AIME25 | 30 | `datasets/aime25/aime_2025.jsonl` | `answer` |
| MBPP | 257 | `datasets/mbpp/mbpp_test.jsonl` | `test_list`，保留转换后的测试集合 |
| HumanEval | 161 | `datasets/humaneval/humaneval-py.jsonl` | `test` / `entry_point`，保留转换后的集合 |
| CodeContest | 165 | `datasets/codecontest/test.jsonl` | `tests`，原转换器无题号，保留行序 |
| LiveCodeBench v1 | 400 | `datasets/livecode/livecodebench_v1.jsonl` | public/private test cases |

这些文件是已有实验使用的导出版本。MBPP 的 257 题和 HumanEval 的 161 题尤其需要原样保留；直接替换成另一来源的默认 split 会改变分母。GSM8K 的上游为 [OpenAI grade-school-math](https://github.com/openai/grade-school-math)，AQuA 的上游为 [Google DeepMind AQuA](https://github.com/google-deepmind/AQuA)，CodeContest 的上游为 [DeepMind code_contests](https://huggingface.co/datasets/deepmind/code_contests)。原始下载 revision 和转换脚本未完整保存在公共资产中，因此复现实验直接采用仓库及服务器保留的导出文件。

### LiveCodeBench 的获取

原导出来自 `livecodebench/code_generation_lite` 的 `release_v1`。官方说明该版本是 2023 年 5 月至 2024 年 3 月的 400 题；其 [官方 loader](https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/main/code_generation_lite.py) 将 `release_v1` 映射到 `test.jsonl`，后续版本另外追加文件。参见 [数据卡](https://huggingface.co/datasets/livecodebench/code_generation_lite)。

`configs/rebuttal_livecode_v1_ids.json` 保存历史 400 题的 platform/question_id 顺序及题目元信息：9 道 Codeforces、181 道 LeetCode、210 道 AtCoder。安装及盘点会校验该顺序。首选原存档；缺少存档时可以从官方仓库获取同一题目集合：

```bash
python scripts/rebuttal/prepare_assets.py fetch-livecode \
  --assets-root /path/to/new/assets --source-revision main
```

该命令只下载 `test.jsonl`，不执行远端 Python loader；下载后检查 400 题身份和顺序。可用 `--source-revision` 记录已选的上游 revision。下载版本使用该 revision 下的题面和测试，历史文件则保留当时的测试内容。若身份不匹配，文件保留为 `.download.jsonl`，不会安装到正式位置。

## 3. 标签从哪里来，需要人工标注什么

| 阶段 | 使用的信息 | 如何获得 |
| --- | --- | --- |
| 离线失败发现 | 训练题、参考答案/参考程序、测试、MAS 轨迹 | 现成训练集；grader 筛失败，数学训练脚本还有 LLM quick check |
| 离线 criterion 挖掘 | 失败轨迹及参考信息 | supervisor 生成诊断定义、触发条件和检查指令，随后语义检索与 LLM 去重 |
| 在线检索与审计 | 当前题目、上下文、agent 输出、固定 bank 中的 criterion 字段 | embedding 检索、selector 选择、auditor 返回二元判断 |
| 最终评分 | 最终回答与评测集参考答案/测试 | 独立 grader；代码执行测试 |
| 审计质量/失败类型研究 | 抽样中间输出和审计判断 | 另做盲评标注，报告标注者数、一致率和分歧处理 |

主方法复现无需新增人工标签。bank 中 `metadata` 保留部分训练参考答案/代码，用于来源追溯；在线 audit prompt 拼接 `detailed_definition`、`trigger_condition` 和 `risk_alert`，参考答案由离线挖掘和最终 grader 使用。新增任务级泄漏检查时同时核对训练/评测题目重合及实际送入模型的请求记录。

## 4. 现有 bank 与重建材料

| Bank | 文件 | 当前数量及来源 |
| --- | --- | --- |
| Math | `pools/math/deduped-mixed_metrics_two_pool.json` | 2,000；metadata 中 1,376 条标为 math、1 条标为 aqua、623 条没有 source_dataset，保留 source_task_id |
| Code | `pools/code/deduplicated_metrics_pool.json` | 2,545；KodCode 745、MBPP train 82、CodeContest sample1000 1,718 |

数学训练题已附在 `train/project_datasets/MATH/processed_math/train.jsonl`（7,500 题）和 `train/project_datasets/aqua/train.jsonl`（97,467 题）。MBPP 训练文件在 `paper/assets/datasets/mbpp/mbpp_train.jsonl`（120 题）。重建代码 bank 还需要服务器归档中的 KodCode、CodeContest 训练采样、失败轨迹及代码专用挖掘脚本；`train/` 的公开收集入口目前覆盖数学和 AQuA。交接时保留原三个代码子池、合并顺序和 `merge_manifest.json`。

当前两个公开 bank 都没有完整 `match_profile.interface_shape` / `io_contract` 字段。代码 runtime 对缺少 profile 的 criterion 使用 `any`；`complete_match_profiles` 会在盘点中显示。研究 profile 过滤效果时，应先生成并保存 profile 扩充版本，再与原 bank 分开比较。

非去重数学池消融另用 `test/metrics_pool/two_pool/mixed_metrics_two_pool.json`，含 2,597 条criterion。服务器交接包同时保存其匹配的 `mixed_embeddings_cache_two_pool.jsonl`（2,597 个4096维向量）。当前release扩展入口使用 `test/metrics_pool/` 下的cache路径，可将三个cache分别链接到对应位置：

```bash
ln -s "$PWD/paper/assets/pools/math/deduped-mixed_two_pool-trigger.jsonl" \
  test/metrics_pool/two_pool/deduped-mixed_two_pool-trigger.jsonl
ln -s "$PWD/paper/assets/pools/code/deduplicated_embeddings-trigger.jsonl" \
  test/metrics_pool/code_mixed/deduplicated_embeddings-trigger.jsonl
ln -s /path/to/preserved/mixed_embeddings_cache_two_pool.jsonl \
  test/metrics_pool/two_pool/mixed_embeddings_cache_two_pool.jsonl
```

已存在的链接保留即可。缺少非去重cache时，使用下面的`embed`命令，将输入改为上述2,597条bank，生成独立cache。

### 直接为现有 bank 生成 embedding

先按 [主实验说明](../paper_main_reproduction.md) 安装 Python 3.10 环境并设置 `EMBEDDING_URL`、`EMBEDDING_MODEL`、`EMBEDDING_KEY`。计划默认只打印；加 `--execute` 后调用服务。每步使用新的输出目录。

```bash
python scripts/rebuttal/build_bank.py embed \
  --input-file paper/assets/pools/math/deduped-mixed_metrics_two_pool.json \
  --output-dir /path/to/new/math-embeddings --execute
python scripts/rebuttal/build_bank.py embed \
  --input-file paper/assets/pools/code/deduplicated_metrics_pool.json \
  --output-dir /path/to/new/code-embeddings --execute
```

脚本沿用已有 embedding 程序：优先编码 trigger condition，缺失时编码 diagnostic definition，输出归一化向量。包装器在结束后核对 bank/cache 名称、维度、有效向量和覆盖率；原程序即使因部分 API 失败而正常退出，覆盖不全仍会使包装器返回失败。

### 从数学/AQuA 训练轨迹重建 bank

设置四组模型服务变量：`SELECTOR_*`、`REASONING_*`、`SUPERVISOR_*`、`EMBEDDING_*`，每组包括 `URL`、`MODEL`、`KEY`。训练使用已有参考标签，不需要逐题人工标注。先移除 `--execute` 查看计划；`--limit 2` 可检查两题链路，正式构建再选择训练规模。

```bash
python scripts/rebuttal/build_bank.py collect-math --limit 2000 \
  --output-dir /path/to/new/math-traces --execute
python scripts/rebuttal/build_bank.py collect-aqua --limit 2000 \
  --output-dir /path/to/new/aqua-traces --execute
python scripts/rebuttal/build_bank.py merge-trajectories \
  --input-file /path/to/new/math-traces/trajectories.json \
  --input-file /path/to/new/aqua-traces/trajectories.json \
  --output-dir /path/to/new/merged-traces --execute
python scripts/rebuttal/build_bank.py deduplicate \
  --input-file /path/to/new/merged-traces/trajectories.json \
  --output-dir /path/to/new/rebuilt-bank --execute
```

合并会给不同来源的 task ID 加独立前缀，避免数学题 0 覆盖 AQuA 题 0。`deduplicate` 的输入必须已有 `generated_adversarial_metrics`；它接续训练收集器的挖掘结果，输出 `bank.json` 与 `embeddings.jsonl`。重建后 criterion 数量由本次失败样本与去重结果决定；比较时记录训练规模、模型、采样设置、原始 criterion 数和去重后数量。

每步输出 `plan.json`、`validation.json`，模型步骤另有 `process.log`。这些文件保存在实验输出目录。keys 仅从环境读入，分享仓库时只提交程序、配置样例和经整理的汇总。
