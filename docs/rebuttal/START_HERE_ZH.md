# 接手实验，从这里开始

本页是给后续实验负责人的单一入口。先核对论文与已有结果，再决定是否运行补充实验；下面的初次检查不调用模型。

## 需要的文件在哪里

| 材料 | 位置 | 获取方式 |
| --- | --- | --- |
| 主方法、机制对照、同库单智能体、采样投票 | 本仓库 `test/`、`paper/runtime/`、`scripts/rebuttal/` | `git clone` |
| 参数矩阵、模型角色、端点样例 | 本仓库 `configs/` | 复制样例后填写当前服务配置 |
| 12个评测文件、数学/AQuA训练数据、两个准则库 | 本仓库 `paper/assets/`、`train/project_datasets/` | 已随代码提供 |
| LiveCodeBench v1、数学/代码向量缓存 | 服务器已保存；也可下载/重建 | [原文件安装、官方下载、缓存生成](data_and_banks.md) |
| 200项准则的双人盲审模板和合并程序 | [标注材料](annotation/README.md) | 下载原始xlsx，分别交两位标注者填写 |
| 论文、原始结果/轨迹、旧程序、来源核查、逐条回复草稿 | 项目私有交接包及原始归档 | 通过项目负责人的服务器访问方式取得 |
| 模型权重与服务 | 实验服务器现有模型目录，或自行部署 | 按[模型角色说明](README.md#安装环境和模型服务)配置 |

公开仓库可用于阅读方法、准备环境及安排新实验。复核历史论文表格，需要私有交接包中的原始运行记录；公开链接不会授予服务器权限。无需为了接手先下载全部历史轨迹。

## 第一次打开仓库

```bash
git clone https://github.com/TonySY2/MASRubric.git
cd MASRubric
python3.10 -m venv .venv-paper
.venv-paper/bin/pip install -r paper/requirements.txt
source .venv-paper/bin/activate

python scripts/rebuttal/prepare_assets.py inventory
python scripts/rebuttal/run_matrix.py list
python test/run_paper_main.py --list
```

全新clone的资产盘点会列出尚未安装的LiveCodeBench文件和两份缓存。按[数据说明](data_and_banks.md)补齐后再用 `inventory --strict` 验收。其余12个测试切分共3,631题，加LiveCodeBench的400题，共4,031题；不要换成名称相同但题数不同的默认数据集。

OlympiadBench另建 `.venv-olympiad` 并安装 `paper/requirements-olympiad.txt`，将解释器通过 `--olympiad-python` 传给实验计划。模型服务使用 `configs/rebuttal.env.example`；端口、GPU和密钥由实际使用者配置。

## 沿用历史结果前，先读取交接包

服务器交接根目录的 `README_HANDOFF_FOR_COAUTHOR.md` 给出实际路径，`handoff.local.json` 保存路径配置。最先打开这些材料：

1. `review-response-20261106/00_处理结果与下一步.md`：当前问题、完成项和后续顺序。
2. `review-response-20261106/truthfulness_corrections_zh.md`：表格与结论更正，特别是PRM聚合、14B Single-Agent来源、评估集参与选参，以及空审计口径。
3. `review-response-20261106/analysis/STATISTICAL_AUDIT_ZH.md`：逐题覆盖、宏/微平均、配对统计和历史重复。
4. `review-response-20261106/manuscript_replacement_text.md`：论文替换文本。
5. `review-response-20261106/responses_current_evidence_en.md`：仅使用已核实证据的逐条英文回复；长稿另保留未完成实验的位置。

论文已有数字、重新核查的结果和未来新实验应分别标记。程序测试通过不代表完成了一套benchmark实验。某些原始记录仍有缺口，具体项目和证据级别已在交接包列明。

## 如果决定继续实验

| 目的 | 程序与说明 | 当前准备状态 |
| --- | --- | --- |
| 完整数学/代码主表及重复 | [运行计划、续跑、汇总](README.md) | 统一入口已实现；历史结果和新运行分开保存 |
| 反馈修订与消息扣留的独立作用 | [机制对照](review_mechanisms.md) | 五种条件与逐轮记录已实现 |
| 单智能体是否同样受益 | [同库单智能体](single_agent_bank.md) | MATH-500/AQuA入口已实现 |
| 增益是否只是增加推理预算 | [预算采样与投票](sampling_controls.md) | MATH-500/AQuA逐题token上限及记账已实现 |
| 准则是否可靠 | [双人独立标注](annotation/README.md) | 模板、合并、分歧裁决工具已备齐；标签为空 |
| 修订纠错/破坏、误拒情况 | [消息版本标注](review_mechanisms.md#double-review-of-revisions-and-false-positives) | 轨迹导出与双人标注汇总已实现；需要实际标签 |
| 新来源准则库、同监督经验基线 | [建库流程](data_and_banks.md#4-现有-bank-与重建材料)、[完整实验清单](experiment_readiness_zh.md) | 数学/AQuA可收集并重建；AutoGuide/ExpeL的公平适配尚需实现 |
| 旧Fixed-MAS及其他历史对照 | [历史程序目录](historical_programs.md) | 原程序在私有归档，使用独立工作副本 |

先用两题检查服务、判题、usage和轨迹，再按预先固定的完整题集与重复次数执行。服务器已有数学主表三次重复（54个任务）和MATH-500/AQuA五种机制三次重复（30个任务）的待运行计划；这些计划尚未启动。

继续前确定一位实验负责人，使用新的输出目录，保存每次结果而不按分数挑选。完成的结果、缺失输出、异常、人工待标注和未运行条件分别报告。全部程序和实验家族详见[操作手册](README.md)，研究内容详见[改写稿解读](manuscript_reading_zh.md)。
