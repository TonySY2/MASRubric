# MASRubric 实验交接入口

接手后先完成环境和资产检查，再跑小样本，最后选择完整实验。已准备的程序包括固定题集安装、失败轨迹收集与建库、主表和受控消融、重复运行、成对统计、训练/测试题面重叠检查，以及审计案例导出。

## 阅读顺序

1. [改写稿解读](manuscript_reading_zh.md)：先理解方法、数据标签的用途和当前表格。
2. [数据与建库](data_and_banks.md)：13 个固定切分、两套 bank、缓存、原始训练数据和获取方式。
3. 本页的运行命令：环境、实验计划、启动、续跑和汇总。
4. [实验准备与证据清单](experiment_readiness_zh.md)：按评审问题选实验，查历史证据的使用条件。
5. [历史程序目录](historical_programs.md)：Fixed-MAS、比较方法、旧消融和分析程序的入口。

## 安装环境和模型服务

```bash
python3.10 -m venv .venv-paper
.venv-paper/bin/pip install -r paper/requirements.txt
python3.10 -m venv .venv-olympiad
.venv-olympiad/bin/pip install -r paper/requirements-olympiad.txt
source .venv-paper/bin/activate
cp configs/rebuttal.env.example configs/rebuttal.env
# 填入实际已部署的四类模型服务，再加载配置。
source configs/rebuttal.env
```

Reasoner 和 auditor 的主配置均为 Qwen3-8B，selector 为 Qwen3.5-9B，embedding 为 Qwen3-Embedding-8B。14B 迁移时将 reasoner 和 auditor 同时指向 Qwen3-14B，库和 selector 保持固定。`math_14b` 是实验标签；实际模型取自环境变量。模型服务的启动由服务器使用者分配 GPU 后进行。

OlympiadBench 使用独立旧版 grader 环境；OlymMATH 使用默认环境中的 math-verify。代码测试会执行生成的 Python，运行账户和容器应只拥有本次实验所需的目录权限。

## 准备资产

```bash
python scripts/rebuttal/prepare_assets.py install \
  --livecode-file /path/to/preserved/livecodebench_v1.jsonl \
  --math-cache /path/to/preserved/math-cache.jsonl \
  --code-cache /path/to/preserved/code-cache.jsonl --strict
python scripts/rebuttal/prepare_assets.py inventory --strict
python test/run_paper_main.py --suite math_8b --benchmark gsm8k --preflight
```

外部资产缺失时，按[数据说明](data_and_banks.md)恢复原文件或生成缓存。不要在同一条对照里更换数据版本、embedding 模型或 bank。完整题集 4,031 题：数学 3,048，代码 983。

## 从 smoke 到完整实验

计划生成和命令检查不调用模型。每份计划使用新的名字和输出目录；运行前可以直接阅读计划 JSON。

```bash
python scripts/rebuttal/run_matrix.py list
python scripts/rebuttal/run_matrix.py plan \
  --family main --suite math_8b --benchmarks gsm8k --limit 2 \
  --plan runs/smoke-plan.json --output-dir runs/smoke
python scripts/rebuttal/run_matrix.py check --plan runs/smoke-plan.json
python scripts/rebuttal/run_matrix.py run --plan runs/smoke-plan.json

# 完整数学主表，两种方法各做三次新运行。
python scripts/rebuttal/run_matrix.py plan \
  --family main --suite math_8b --repeats 3 \
  --olympiad-python "$PWD/.venv-olympiad/bin/python" \
  --plan runs/math-main-3repeats.json --output-dir runs/math-main-3repeats
python scripts/rebuttal/run_matrix.py check --plan runs/math-main-3repeats.json
python scripts/rebuttal/run_matrix.py run --plan runs/math-main-3repeats.json
```

代码主表使用 `--suite code_8b`；14B 使用 `--suite math_14b`。每题的 repeated run 保留真实重复编号，现有模型接口未建立采样 seed 契约，因此不把 repeat 编号写成模型随机种子。

实验按计划顺序执行，默认单题并发为 1、每个实验超时为 24 小时。`--timeout` 可显式设置任务上限，`--keep-going` 可在一个实验失败后继续其余任务。`--resume` 跳过已完成且通过回读检查的任务。失败或部分完成的目录保留原状；重新运行该项时生成新计划和新输出目录。

## 已配置的实验家族

| `--family` | 变化内容 | runtime |
| --- | --- | --- |
| `main` | baseline / MASRubric，数学 8B/14B、代码 8B | reference |
| `revision` | 重写预算 0/1/2/3/4；0 保留审计与扣留 | reference |
| `rubric_size` | Kctx 3/5/7，阈值固定 0.6 | reference |
| `threshold` | 阈值 0.4/0.6/0.8/1.0 | reference |
| `candidate_size` | 候选数 10/20/40，Kctx 上限 5 | reference |
| `retrieval_control` | contextual / 固定 5 条随机项 / generic audit | reference |
| `code_sensitivity` | 代码 direct top-1/3/5、阈值 0.6/1.0 | reference |
| `fixed_extension` | 当前固定 FullGraph baseline / MASRubric | configurable |
| `pool_control` | 去重库 / 未去重库 | configurable |
| `random_1to5_extension` | contextual / 随机 1–5 项 | configurable |

例如，只跑 AIME24、MATH-500 的 Kctx 对照：

```bash
python scripts/rebuttal/run_matrix.py plan \
  --family rubric_size --suite math_8b --benchmarks aime24,math500 \
  --repeats 3 --plan runs/kctx.json --output-dir runs/kctx
```

主表默认参数保持原样。单次受控参数覆盖必须命名，参数会写入 manifest：

```bash
python test/run_paper_main.py --suite math_8b --benchmark aime24 \
  --variant-name kctx3 --select-q 3 --pass-rate 0.6 --retry-budget 3 --dry-run
```

历史 Fixed-MAS 表、其他比较方法、等预算停止规则和新诊断协议分别见[历史程序目录](historical_programs.md)与[实验清单](experiment_readiness_zh.md)。每条实验均应在同一 runtime 内比较 control 与 treatment。

## 结果、统计和诊断

```bash
python scripts/rebuttal/summarize.py runs/math-main-3repeats \
  --output runs/math-main-summary.json

python scripts/rebuttal/summarize.py --paired \
  /path/to/control/result.json /path/to/treatment/result.json \
  --output runs/paired-comparison.json
# 旧 AQuA 仅保存选项结果时，增加 --benchmark aqua。

python scripts/rebuttal/audit_traces.py /path/to/result.json \
  --output-dir runs/audit-inspection --sample 100

python scripts/rebuttal/check_split_overlap.py \
  --train /path/to/train.jsonl --test paper/assets/datasets/math500/test_ids.jsonl \
  --output runs/math-overlap.json
```

摘要保留完整分母、已记录、正确、错误、缺失、未知和 usage 完成性。成对比较按题目 ID 对齐，输出差值、bootstrap 区间和 McNemar 检验，同时列出未配对和未知样本。`suite_repeats` 按模型、runtime、family、方法和 repeat 分组，只为完整的 9 项数学或 4 项代码计算宏平均；缺项、重复项、错误分母和 smoke 子集单列，宏平均留空。

`audit_traces.py` 统计最终消息 score 记录；全部审计调用需要完整 trace。缺失或格式错误的 judgment 与实际空列表分别计数。`annotation_blind.json` 可交给独立核对者；对应模型判断保存在 `annotation_key.private.json`，核对完成后再合并。需要上游上下文才能判断的消息，在 `requires_additional_context` 中标记。建库和主表运行使用数据集现成标签，无需逐条人工写 criterion；该标注表用于额外衡量 auditor 的误判率。

题面重叠工具比较显式提供的训练/测试导出，使用 Unicode 和空白归一化后的精确文本。题面改写和模型预训练暴露需要另外的研究设计。

## 交接文件

公开仓库保存源码、配置、题集身份清单和操作文档。服务器侧另存原始结果、训练数据、完整轨迹、原始论文、历史代码与环境清单。项目负责人向接手者提供该私有资产目录和访问方式；大模型权重使用已存在的模型目录。

每次实验的 `job.json`、runner manifest、原始 result、grader、usage、日志和最终 summary 保存在独立目录。人工整理的论文表格与机器生成的实验结果分别存放。
