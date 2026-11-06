# Rebuttal 实验准备与证据清单

本清单按 MASRubric 的论文主张组织实验。先准备资源、程序和配置，再根据实际评审问题选择需要执行的任务。

## 先建立可用的运行基线

1. 按[主方法运行说明](../paper_main_reproduction.md)建立 Python 3.10 运行环境和独立 OlympiadBench 判题环境。
2. 准备固定的 13 个测试切分、两个 criterion bank、对应 embedding cache；LiveCodeBenchV1 必须是论文的 400 题输入。
3. 用 `test/run_paper_main.py --list` 核对任务，用 `--dry-run` 查看命令，用 `--preflight` 检查数据、缓存、模型角色和判题器。
4. 先跑每类一个真实样本，核对模型服务、输出格式和 grader，再启动完整任务。smoke 子集保留为 smoke，完整表单独运行。
5. 每次运行保存配置、题目 ID、结果、usage、审计轨迹、失败状态和完成数；两种方法使用同一题目集合。

主方法入口覆盖 Dynamic-MAS 的 `math_8b`、`math_14b`、`code_8b`，各有 `baseline` / `masrubric`。`test/run_release_experiment.py` 是另一套可配置 runtime。两者的实现差异和论文对应关系见[实验矩阵](../experiment_matrix.md)。

## 实验目录

“现有入口”表示有程序可执行；论文结果是否逐项复现，需要对应的完整运行结果。需要增加控制开关的项目列明所需改动，避免用名称相似的 preset 替代目标实验。

| 优先级 | 实验 | 固定与变化项 | 需准备的程序或资产 | 核心产物 |
| --- | --- | --- | --- | --- |
| P0 | Dynamic-MAS 数学 8B 主表 | 9 个固定测试切分；baseline / MASRubric | 主方法入口、数学库与缓存、两个 grader 环境 | 每题结果、9 项准确率与宏平均、完成分母 |
| P0 | Dynamic-MAS 代码 8B 主表 | 4 个固定测试切分；baseline / MASRubric | 主方法入口、代码库与缓存、400 题 LiveCodeBench、隔离执行环境 | 重新执行测试后的 pass 结果、超时/异常计数 |
| P0 | 14B 迁移 | 库和全部推理参数固定；替换 backbone | `math_14b` 入口、14B 模型服务 | 准确率、audit size、repair rounds、拒绝率 |
| P0 | 独立重复与统计 | 预先固定成对重复计划，同时重复两种方法 | 跨完整 benchmark 的 run manifest、每题配对统计、置信区间脚本 | 每次原样宏平均、均值/标准差、成对差值 |
| P0 | 数据、标签与切分核对 | 固定数据版本、题目 ID、构库输入 | 数据清单、训练/测试重叠检查、在线请求采样记录 | instance overlap、split 关系、grader 与在线输入边界 |
| P0 | 完成性与失败处理 | 缺失输出、重复 ID、解析错误、API 错误 | launcher 完成性检查、审计异常计数 | 成功/错误/缺失/未知分列及退出状态 |
| P1 | Fixed-MAS 主表 | 固定论文 DAG、角色、prompt、调用顺序 | 历史 Fixed-MAS 源码与原配置的对应关系 | 51.15→52.74 对应行的可追溯运行 |
| P1 | 主表比较方法 | Single / CoT / Self-Refine / PRM / Multi-TAG / RaR-Gen | 各历史 runner、prompt、模型和停止规则；PRM 权重 | 每方法原始结果及成本；相同题目集合 |
| P1 | 最大重写轮数 | `Tmax=1,2,3,4`；其余保持默认 | 主方法参数覆盖或匹配的历史 runner | Table 3 Block I、总 token、纠正/破坏计数 |
| P1 | rubric 数量 | `Kctx=3,5,7`；数学阈值固定 0.6 | 支持独立覆盖 Kctx 的入口 | Table 3 Block II、实际 rubric size 分布 |
| P1 | 通过阈值 | `0.4,0.6,1.0`；Kctx 上限固定 5 | 阈值独立覆盖 | Table 3 Block III、接受/重写/拒绝分布 |
| P1 | 去重消融 | 固定原始候选池和构库模型；有/无去重 | 两套库、两套完整缓存、构库日志 | Table 3 Block IV、库大小、覆盖与成本 |
| P1 | 检索/通用项对照 | retrieved / random 1–5 / generic | 历史配置或核对后的 release presets | Table 3 Block V、每条消息实际检查项 |
| P1 | 等数量随机检查项 | 逐消息固定检查项数与预算，随机选内容 | 记录数量并重放的控制入口 | 检索内容贡献与检查项数量贡献分离 |
| P1 | 在线成本与串行延迟 | 所有方法统一题集、并发、硬件和 warmup | usage 分角色汇总、时钟记录、服务配置 | token/calls、p50/p95 单题延迟、吞吐 |
| P1 | 离线构库成本 | rollout、miner、dedup、embedding 分阶段 | 训练轨迹、usage 日志、保留/删除计数 | 每阶段成本、累计 token、构库完成状态 |
| P1 | 运行行为图复核 | 同一主实验输出集合 | audit/selection/repair trace 统计脚本 | Figure 3–5、Table 6–7 的可重算表 |
| P1 | 审计 verdict 有效性 | 冻结中间消息；独立核对错误与适用性 | 固定样本、审计输出、可执行判定或独立标注表 | precision/recall、false pass/reject、覆盖率 |
| P1 | 修正与 withholding 归因 | full / revise-only / withhold-only / audit-only | `review_mechanism` 统一入口；逐轮 trace 保留选择、修正、传播与 fallback | 正确→错误、错误→正确、最终成功率 |
| P2 | 字段与表示消融 | definition / applicability / directive 的受控删减 | 对应 bank 视图、检索和审计输入开关 | 哪个字段支持检索、诊断与修正 |
| P2 | 检索策略 | direct / rerank；Kcand；embedding 表示 | 同一 runtime 的独立开关和缓存 | 检索质量、最终准确率、选择开销 |
| P2 | bank 来源与规模 | 固定 teacher 后改变规模，或等规模改变 teacher | 可追溯子池和重算缓存、固定采样规则 | 来源效应与规模效应分开 |
| P2 | 留一来源/领域迁移 | 去掉某训练来源；测试集合保持不变 | 按训练来源归属的 bank lineage | 同分布、跨分布、留一来源结果 |
| P2 | Reasoner 与 auditor 分离 | 8B/14B reasoner × 8B/14B auditor；selector 固定 | 分角色 endpoint 配置、统一上限 | backbone 能力与审计能力分别的贡献 |
| P2 | 预算匹配曲线 | 统一 token/call/time budget；多档预算 | MATH-500/AQuA 的 `sampling_vote.py` 支持总 token 上限；调用数和时间预算另设协议 | accuracy–budget 曲线与成本 Pareto 前沿 |
| P2 | 难度信号检验 | 冻结 audit depth、withholding 定义 | 逐题聚合与 held-out 统计脚本 | 与错误率的相关性、分组及跨模型稳定性 |
| P2 | 误判与过度修正案例 | 使用主实验逐消息轨迹 | 前后答案解析、原题 grader、人工复核导出 | 可定位案例和总体发生率 |

P0 是实验可交接和结果可解释的基础；P1 覆盖论文现有主张；P2 用于有针对性的机制和泛化追问。

## 需要对齐的实现细节

- 论文写 6 个 chat turns。恢复 runtime 的 message limit=7 包含初始 user 消息和至多 6 条 agent 消息。
- 数学 final-answer temperature 为 0.7；恢复的代码 final 类显式使用 0.0。论文 §4.1 将非 auditor 温度统一写为 0.7，需要保留这项差异的实验记录。
- 恢复代码路径使用 direct top-3 retrieval；论文 §3.3 描述 coarse-to-fine selection。比较两者应单独命名并记录。
- release 的 `masrubric_math_top3` 同时将阈值设成 1.0。论文 Kctx=3 消融保持默认阈值 0.6，不能直接用该 preset 代替。
- batch audit 会看到 definition、condition 和 directive；稿件 judgment 阶段写 condition 与 directive。字段消融应固定其他输入。
- 空判断集可能通过原 runtime 的接受分支。需要分别记录无适用项、解析失败和 API 失败，避免将它们并入正常通过率。
- Dynamic runtime 在剩余可用消息过少时会重启为无干预任务。保留 reset 次数、触发条件和回退输出归属。
- code bank 没有填充的 `match_profile`。打开 profile-aware 选项本身不能产生有意义的 profile 过滤。

## 统计与结果保存

### 一次结果的最小记录

保存方法、runtime、模型角色与实际服务模型名、配置、题集标识、题目 ID、随机种子/重复编号、起止时间、并发、完整/部分状态、每题输出和 grader 结果。保存每次审计选出的 criterion ID、判断、反馈、重写次数、最终发送/扣留动作及 API/解析异常。usage 缺失记为 unknown，不能补成零。

### 分母

| 统计 | 分母 |
| --- | --- |
| 论文准确率 | benchmark 固定测试题数；缺失输出仍保留在分母中 |
| 数学/代码宏平均 | 完整的 9/4 项 benchmark 准确率 |
| rubric size | 逻辑 audit 次数 |
| 首轮通过、修正后通过、withheld | 被拦截的 agent output 数 |
| problem reset | Phase-1 problem execution 数 |
| 串行延迟 | 明确列出的题目及同一并发配置 |

历史 8B 日志的 18,270 个 agent outputs、22,443 次 audits 和 3,046 次 Phase-1 executions 是不同分母；14B 的 outputs 为 18,282，executions 为 3,048。论文完整数学测试集仍为 3,048 题。不能把这几个数互相替代。

### 成对重复

每个 repeat 保存整套方法结果，先逐 benchmark 配对，再汇总宏平均。不能逐 benchmark 挑出最好的一次，拼成一个新 repeat。每题配对 bootstrap 或 McNemar 可用于单个 benchmark；跨 benchmark 的结论还应保留每项差值和宏平均结果。小题集同时给出正确题数。

## 历史证据怎样复用

| 历史材料 | 可复用内容 | 重算或补证据要求 |
| --- | --- | --- |
| 主表原始结果与逐任务 summary | 原结果、题目 ID、原配置、判题与 usage | 与当前稿逐行对应，并核对完整分母 |
| 7 月方差 V3 | 历史逐任务结果和运行 manifest | V3 前两列存在逐任务后验挑选/拼接，不能作为两次独立重复。四次原始独立重复的数学宏平均记录为 52.8498 / 50.2737 / 52.0786 / 51.8456，均值 51.7619、样本标准差 1.0810pp；相对 baseline 的方差还需要成对重复 |
| 7 月 Layer V3 | 五个 run family 的完成结果 | 九项来源跨不同 repeat；与当前稿 Fixed-MAS 行分别对应，统一配置重跑时使用预先固定整套计划 |
| Gemma 构库实验 | 新来源库与运行产物 | teacher、dedup、内容、规模同时变化（2904 vs 2000）；teacher-only 实验需要固定其他因素 |
| 7 月过度修正案例 | 现象样例及其完整轨迹 | 原样例来自 MAD 轨迹；主方法总体统计应重新提取自身轨迹 |
| 7 月延迟记录 | concurrency=1 的九个固定 probe；另有 concurrency=32 的吞吐摊销 | 147.48s 是九个 probe 平均；完整测试集串行 latency 需要完整采样计划，不能用 wall_time/completed 直接替代 |
| 7 月离线数学池成本材料 | 原始逐调用 usage、calls、运行来源 | rollout 统计为 4,000 次尝试、3,958 次完成、52,153 calls、约 97.13M tokens；Gemma 构库的实测生成/去重 token 为 8,564,936 / 9,852,512，保留 2,904 条检查项 |
| 14B 运行统计 | 检查项数、修正深度、拒绝和 reset | 固定同一库、分别保存模型角色；共享上限的观察不能代替逐模型调参实验 |

成本来源补充：逐行求和确认 Gemma 生成的 2,868 条调用及去重的 5,789 条 decision 均有完整 usage。把两阶段实测 token 分别乘以 `2000/2904`，得到 5.899M 和 6.785M，四舍五入与 A.6 的 5.90M、6.79M 一致。这提示按库规模归一化的可能来源；尚未找到明确采用该公式的记录。原始 manifest 记录的生成与去重模型均为 `gemma-4-12b-it`，因此这些数值不能直接作为稿件所述商业 API 构库的实测 usage；对应模型的原始调用记录或明确的估算方法仍需补齐。

当前论文主张的证据依据是对应运行与可重算统计。历史过程稿、准备好的 manifest、未启动任务和 smoke 记录保留各自状态。

## 交接时的阅读顺序

1. [改写稿解读](manuscript_reading_zh.md)：确定问题、机制和表格对应。
2. [主方法运行说明](../paper_main_reproduction.md)：环境、数据、服务与执行命令。
3. [实验矩阵](../experiment_matrix.md)：区分主方法 runtime、扩展 preset 和其他方法。
4. 本清单：按实际评审问题选择实验，先检查输入、原始证据和统计分母，再安排计算资源。
