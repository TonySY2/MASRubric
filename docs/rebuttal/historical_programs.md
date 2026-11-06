# 历史实验程序索引

当前主实验优先使用 `test/run_paper_main.py`；当前新增对照使用本目录的实验矩阵。完整旧程序保存在服务器归档中，下面给出单任务入口、依赖输入和可接续的分析工具。公开仓库的 `configs/rebuttal_legacy_programs.json` 记录了 21 个已检查程序及其参数名。

## 目录与只读检查

`MASRUBRIC_ARCHIVE_ROOT` 指向服务器交接包的 `server139` 归档根目录。其下的三个源码根目录为：

| 代号 | 归档相对路径 | 用途 |
| --- | --- | --- |
| AC | `july-rebuttal/reference_packages/main_method_reference_20260711/source_workspace` | Dynamic-MAS、数学/代码主实验、消融、CoT、MAD、采样与分析 |
| FIXED | `historical-projects/26.3.24rebuttal/V2forV1` | 历史固定图框架、动态生成审计条件、离线bank版、PRM服务 |
| TRAIN | `historical-projects/26.4.9开始-改进V2/metric_training_workspace` | 代码失败轨迹挖掘及去重流程 |

```bash
export MASRUBRIC_ARCHIVE_ROOT=/path/to/handoff/private-assets/server139
python scripts/rebuttal/inspect_legacy.py --strict
```

路径不同时可单独提供 `--ac-root`、`--fixed-root`、`--training-root`。检查器只读源码并解析 argparse，不执行 import、`--help`、服务启动或实验。它会报告缺文件、语法错误、参数缺失和损坏的旧脱敏参数。

新运行使用一个独立、可写的源码副本作为 `LEGACY_ROOT`，把 dataset、bank、cache 和模型路径显式传入。历史 runner 会向自身 `results/` 写入，有的还会清理同名运行结果，因此每次给新的 `--exp-name`，保留原归档不动。

## Dynamic-MAS 和单agent对照

以下路径相对于 AC。

| 实验 | 入口 | 关键选项/输入 |
| --- | --- | --- |
| 数学 MASRubric / Dynamic baseline | `scripts/run_v2_math_task.py` | `--task`、`--exp-name`、四组模型、bank/cache、`--dataset-path-override`；baseline加`--baseline-only` |
| 代码 MASRubric / Dynamic baseline | `scripts/run_v2_code_task.py` | 同上；数据参数为`--dataset-path`；代码主配置`--force-direct-search --direct-k 3 --pass-rate 1.0` |
| 数学 Single Agent | `scripts/run_v2_math_task.py` | `--mode vanilla`，实际选用任务对应`run_*_vanilla.py` |
| 代码 Single Agent | `scripts/run_v2_code_vanilla_eval.py` | `--task --in_file --out_file --log_file --reasoning_url --reasoning_model` |
| CoT / MAD | `scripts/run_cot_mad_task.py` | `--domain math/code --task --method cot/mad --exp-name --reasoning-url --reasoning-model --dataset-path` |
| Self-Refine | 两个`run_v2_*_task.py` | `--agent-mode self_refine` |
| Multi-TAG | 两个`run_v2_*_task.py` | `--agent-mode multi_tag` |
| PRM / best-of-N | 两个`run_v2_*_task.py` | `--agent-mode prm --prm-url <服务根地址> --prm-n-samples 3` |

PRM必须提供实际 reward 服务；该实现未提供外部PRM地址时会走LLM候选选择，实验标签要随实际模式记录。历史旧runner会保留较多调用参数与详细日志，建议使用可信的本地服务端点，认证服务通过当前公共入口传入keys。

下面是一个已按源码参数核对的数学单任务模板。先配置环境中的模型名和端点，以及 `LEGACY_ROOT`、`ASSETS_ROOT`、`PAPER_PYTHON`，在独立源码副本中运行：

```bash
cd "$LEGACY_ROOT"
export PYTHONPATH="$LEGACY_ROOT"
export AGENTDROPOUT_MATH_TEAM_VARIANT=verifier
export AGENTDROPOUT_V2_SELECTOR_API_KEY=EMPTY
"$PAPER_PYTHON" scripts/run_v2_math_task.py \
  --task gsm8k --exp-name rebuttal_math_smoke_r1 \
  --python-bin "$PAPER_PYTHON" \
  --reasoning-url "$REASONING_URL" --reasoning-model "$REASONING_MODEL" \
  --supervisor-url "$SUPERVISOR_URL" --supervisor-model "$SUPERVISOR_MODEL" \
  --selector-url "$SELECTOR_URL" --selector-model "$SELECTOR_MODEL" \
  --embedding-url "$EMBEDDING_URL" --embedding-model "$EMBEDDING_MODEL" \
  --dataset-path-override "$ASSETS_ROOT/datasets/gsm8k/gsm8k_id.jsonl" \
  --metric-pool-file "$ASSETS_ROOT/pools/math/deduped-mixed_metrics_two_pool.json" \
  --embedding-cache-file "$ASSETS_ROOT/pools/math/deduped-mixed_two_pool-trigger.jsonl" \
  --retrieve-p 20 --select-q 5 --pass-rate 0.6 --retries-times 3 \
  --batch-audit-metrics --max-turns 7 --concurrency-limit 1 --limit 2
```

使用真实服务验证这两题的prompt、judgment和结果后，再去掉`--limit`。Olympiad使用独立grader解释器。历史任务名中的`olymMATH_easy`/`olymMATH_hard`保留原大小写；公共主入口使用`olymmath_easy`/`olymmath_hard`。

## 消融与扩展

| 研究问题 | 实际程序/开关 | 需保留的控制条件 |
| --- | --- | --- |
| 修订预算 | 数学`run_v2_math_task.py --retries-times 1/2/3/4` | 相同题目、bank、阈值、模型 |
| rubric条数 | `--select-q`，direct模式为`--direct-k` | 阈值单独记录；历史top3 preset同时改阈值 |
| 阈值 | `--pass-rate 0.4/0.6/1.0` | 实际选中criterion数也要保存 |
| 随机检索 | 数学`--random-k-min 1 --random-k-max 5`；代码`--random-k` | 同一bank、相同评测题目 |
| rerank / direct | 数学`--retrieve-p --select-q` vs `--force-direct-search --direct-k` | embedding、criterion内容及revision配置 |
| 是否重新选rubric | 数学`--lock-metrics-after-first-round` | 审计、修订及token开销 |
| 无bank / 通用审计 | 代码`--use-simple-audit 1`；当前release的generic preset | 记录实际prompt，区别于关闭审计 |
| 非去重bank | `--metric-pool-file`和匹配的`--embedding-cache-file` | 原池条目与去重池条目、在线token |
| 跨模型 | 显式替换reasoning/supervisor的model与URL | 其余模型角色和服务设置 |
| Profile检索 | `prepare_profileaware_metric_generation.py` → 标注chunk → `merge_profileaware_metric_profiles.py` → `build_profileaware_embeddings.py` | profile schema、所有criterion覆盖、匹配文本策略 |
| AIME26 / APPS / TACO扩展 | `fetch_aime26_dataset.py`、`prepare_apps_competition_dataset.py`、`prepare_taco_dataset.py` | 新benchmark独立报告，不并入原13项平均 |

Profile工具的旧流程会生成chunk与合并manifest；chunk输出需要LLM/人工完成后再合并。原脚本包含其自身历史校验流程和覆盖输出行为，使用新工作目录。当前bank缺profile时先看 [数据说明](data_and_banks.md)，不要直接将原cache当作profile-enriched cache。

## 历史 Fixed-MAS 与生成式审计

以下路径相对于 FIXED。三种数学入口各自保留，便于核对论文表格的具体来源。

| 实验 | 入口 | 必要输入与关键参数 |
| --- | --- | --- |
| Fixed baseline / agent-mode对照 | `experiments/run_math_benchmark.py` | `--domain --dataset_json`；`--mode FullConnected --agent_nums --num_rounds --train_limit --eval_limit`；agent-mode、PRM参数在迁移版中存在 |
| Fixed生成式审计/修订 | `experiments/run_math_benchmark_v2forv1.py` | `--domain --dataset_json --train_dataset_json`；reasoning/supervisor URL/model、`--retries-times --pass-rate --reject-failed-outputs` |
| Fixed离线criterion bank | `experiments/run_math_benchmark_v2forv1_metric.py` | 上一行加embedding URL/model、`--metric-pool-file --embedding-cache-file --retrieve-p --select-q --batch-audit-metrics` |
| Fixed代码基线 | `experiments/run_humaneval.py` | `--task`选择代码任务，`--dataset_json`、`--train_dataset_json`、agents、rounds、输出文件 |
| Qwen2.5-Math-PRM服务 | `scripts/serve_qwen25_math_prm.py` | `--model --host --port --device --dtype --max-length`，另配PRM环境与空闲GPU |

固定图入口还保留原pruning/dropout/optimization选项。要复查固定拓扑主表时，从对应原run的config读取这些选项与训练题使用方式，不仅设置`FullConnected`。旧`--seed`是否传到模型采样链也须按实际请求记录判断。

RaR-Gen的论文标签目前没有对应同名程序；服务器已有生成式审计入口及原始表格/配置。先从该行的原run config定位其实际入口和prompt，再建立RaR-Gen映射。不能仅因程序名字含reflection就当作该baseline。

## 代码bank重建与结果分析

| 阶段 | 根目录与程序 | 输入/输出 |
| --- | --- | --- |
| 代码训练输入 | AC `scripts/prepare_offline_code_train_pool.py` | 原MBPP train路径、CodeContest上游；`--out-dir --sample-size 1000 --seed 20260430 --shards` |
| 原失败收集 | AC `scripts/launch_offline_train_pool_baseline.sh` | 训练输入、已配置模型服务、轨迹输出；旧批处理含机器路径 |
| 汇总失败 | AC `scripts/finalize_offline_code_train_pool.py` | `--method-dir`，可选`--aggregate-output` |
| 挖掘criterion | TRAIN `scripts/run_oldparadigm_metric_generation.py` | 已保存失败轨迹；`--model --mbpp-batch-size --codecontest-batch-size --base-cap --remaining-cap --max-attempts` |
| 子池去重 | TRAIN `scripts/run_oldparadigm_dedup_dataset.py` | `--dataset --threshold 0.65 --top-k 15`；该脚本可自行启动embedding GPU，先适配服务/资源配置 |
| 合并代码池 | TRAIN原合并记录与`merge_manifest.json` | KodCode 745、MBPP 82、CodeContest 1718，保留来源和原名称 |
| 动态结果CSV | AC `scripts/export_v2_launch_csv.py` | `--exp-prefix`或`--exp-glob`、`--output`，扫描副本的`results/` |
| 审计轨迹可读化 | AC `scripts/render_atomic_trace_log.py` | `--trace-file --output-log`，可选`--summary-json` |

原pool准备脚本中的MBPP路径、训练workspace路径和自动服务启动配置需要映射到接管机器。现成bank直接使用当前 [bank包装器](data_and_banks.md#4-现有-bank-与重建材料)；全流程重建时另保存训练采样manifest、失败数、挖掘模型、去重数量和子池合并记录。

旧`launch_*`、`watch_*`脚本包含端口、GPU、队列接续与服务管理逻辑，适合作为原配置来源。新增运行使用单任务入口或当前矩阵，不直接重启旧批量调度器。原配置、结果、stdout、usage与运行日期一起保留，新的测量写入独立目录。
