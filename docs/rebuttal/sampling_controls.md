# 单 agent 采样与投票对照

`scripts/rebuttal/sampling_vote.py` 在显式总 token 上限下重复生成答案，再做不读取标准答案的投票。当前支持 MATH-500 与 AQuA。所有请求只包含题目和可选选项，标准答案仅在生成结束后的判题阶段读取。

每次生成都计入 prompt 和 completion tokens。运行前通过服务的 `/tokenize` 取得实际聊天模板下的输入 token 数，为下一次输入预留预算；输出上限为剩余预算与单次上限中的较小值。缺少 usage、tokenizer 与实际输入计数不一致、服务超出请求上限时停止，并保留已有回答。服务需要兼容 OpenAI chat API 和 vLLM 的 `/tokenize`。

## 小样本与完整运行

```bash
export REASONING_URL=http://127.0.0.1:8000/v1
export REASONING_MODEL=Qwen3-8B
export REASONING_KEY=EMPTY

# 不带 --execute 时只打印计划。
python scripts/rebuttal/sampling_vote.py run \
  --benchmark math500 --ids 0,1 --total-token-budget 4096 \
  --max-samples 3 --max-output-tokens 1024 \
  --output-dir runs/sampling-smoke --execute
python scripts/rebuttal/sampling_vote.py summarize \
  --run-dir runs/sampling-smoke --output runs/sampling-smoke-summary.json

# 完整集合省略 --ids；使用不同 seed 和新目录作独立重复。
python scripts/rebuttal/sampling_vote.py run \
  --benchmark math500 --budget-file /path/to/frozen-budgets.json \
  --seed 20261106 --max-samples 32 --max-output-tokens 4096 \
  --output-dir runs/sampling-full-r1 --execute
```

匹配目标方法成本时，先冻结其逐题完整 usage，再准备如下预算文件；缺失的 usage 不能补成零或用其他题的成本代替。

```json
{
  "source_run": "target-method-run-identifier",
  "scope": "total_prompt_plus_completion",
  "budgets": {"0": 12000, "1": 16000}
}
```

预算应涵盖目标方法所有模型角色和重试、回退调用。若 embedding 成本另列，应对两种方法一致披露。实验表同时报告预算上限、实耗、未用预算与生成次数，不能把预算上限写成已消耗 token。多档预算在同一冻结题集上比较，每档使用新目录。

## 投票和统计

投票提取最后一个完整 `\boxed{...}`，归一化空白、常见 LaTeX 格式和简单有理数；相同答案票数最高者胜出，并列选最早生成的答案。AQuA 统一选项字母大小写。该规则使用归一化后的精确答案，不执行一般符号等价聚类。无法解析的回答仍保留在生成次数和成本中。

汇总保留完整选定分母、缺失/失败记录和每题结果。MATH-500 使用当前 reference `MathGrader`，AQuA 比较选项。投票准确率与“生成集合中存在正确答案”分别统计。`generated_set_subset_success_at_k` 是从已生成集合均匀抽取 k 个候选时包含正确答案的概率；预算停止和末次截断会改变候选分布，因此不能将它解释成固定次数独立采样的无偏 pass@k。候选不足 k 时该题不可用，整套平均值留空并列出覆盖数。

新运行记录 seed 参数和服务返回的模型名。预算对照的结论需要完整题集、预先固定的重复计划，以及与目标方法一致的判题口径；两题小样本只检查运行和记账链路。
