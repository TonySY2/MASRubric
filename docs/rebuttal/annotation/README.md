# 准则双人独立复核

本目录提供200条准则的空白复核模板，数学和代码各100条。两位标注者分别复制同一份工作簿，独立填写后由协调者合并、组织分歧裁决。

| 文件 | 使用者 | 用途 |
| --- | --- | --- |
| [独立盲审模板](MASRubric_准则独立盲审模板.xlsx) | 两位标注者，各持一份 | 阅读准则，填写四个标签和理由；协议页填写姓名、日期及完成确认 |
| [原文参考表](rubric_review_reference.json) | 协调者及合并程序 | 校验ID、领域和准则原文，只有工作簿可见内容及工作表字段位置 |
| [合并程序](../../../scripts/rebuttal/merge_rubric_reviews.py) | 协调者 | 读取两份xlsx，输出完成状态、分歧清单及分领域统计 |

工作簿包含`name`、`definition`、`condition`、`directive`原文，800个标签均为空。参考表不包含来源任务、源文件索引或模型判断。交给标注者的文件只需xlsx。

## 标注者操作

1. 将模板另存为自己的工作文件，例如`review_A.xlsx`或`review_B.xlsx`。独立填写期间不查看另一位的标签。
2. 在“盲审”页G–J列为每条准则的四个维度选择“是 / 否 / 不确定”。具体定义见工作簿“协议”页。任一维度为“否”或“不确定”时，在K列注明维度及理由。
3. 留空表示未完成；无法根据准则文本作可靠判断时选“不确定”。保留A–F列原文和ID。可整行排序，但不要只排序部分列。
4. 完成200条全部四维标签及必要理由后，在“协议”页B3填姓名、B4填日期（Excel日期或`YYYY-MM-DD`）、B5选“是”。保存并将个人文件交给协调者。

## 协调者操作

使用Python 3，仅依赖标准库。以下命令从仓库根目录执行，填写文件与统计结果放在仓库外的工作目录：

```bash
python scripts/rebuttal/merge_rubric_reviews.py \
  --review-a ../rubric-reviews/review_A.xlsx \
  --review-b ../rubric-reviews/review_B.xlsx \
  --out ../rubric-reviews/merge_round1
```

PowerShell可将上述命令写成一行。程序默认读取本目录的`rubric_review_reference.json`，也可用`--reference`指定同格式参考表。程序只读取xlsx，不修改个人文件，并拒绝重复/缺失/未知ID、非法标签和准则原文改动。

输出包括：

- `review_statistics.json`：完成状态，各领域各维度的数量、分母、一致率和最终可靠率。
- `merged_reviews.csv`：保留两人的原标签与最终标签。
- `adjudication_required.csv`：尚未裁决的分歧。
- `STATISTICS.md`：简明汇总。

两人完成后，裁决者在分歧CSV填写裁决标签、理由、姓名和日期，每项“裁决完成”填“是”。允许裁决为“不确定”，保留原始A/B标签。使用新输出目录再次合并：

```bash
python scripts/rebuttal/merge_rubric_reviews.py \
  --review-a ../rubric-reviews/review_A.xlsx \
  --review-b ../rubric-reviews/review_B.xlsx \
  --adjudication ../rubric-reviews/merge_round1/adjudication_required.csv \
  --out ../rubric-reviews/merge_round2
```

程序要求两份不同的个人文件及不同的标注者姓名。完成确认、姓名、日期、全部标签及必要理由共同决定完成状态。已有输出不会被覆盖；裁决记录保留在统计JSON中。

## 统计定义

数学和代码分别报告，每领域100条：

- 每维度列出“是、否、不确定、未填/未裁决”数量。确定项通过率为`是 / (是 + 否)`，同时报告分母与不确定数量。
- 三分类一致率以两人均已填写的该维度项目为分母，包含两人共同的“不确定”；另报双方均为是/否时的一致率及分母。
- 整体可靠率为“四维均为是”的准则数，除以“四维均有确定最终标签”的准则数。含不确定或未裁决者单列。

两人尚未明确完成时，一致率和通过率为`null`；所有分歧尚未裁决时，最终可靠率为`null`。没有确定项时也为`null`。空白、不确定和未裁决都不会自动计为否。

这些统计描述抽样准则的文本质量；消息纠错率和下游答案变化需要另行标注轨迹。

## 本地检查

```bash
python -m unittest discover -s tests -p test_rubric_reviews.py -v
```

测试覆盖模板原文及空标签、显式完成确认、未知标签分母、分歧裁决、重复标注者，以及重复/缺失ID和原文改动。
