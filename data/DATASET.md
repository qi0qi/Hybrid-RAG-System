# 问答训练集与测试集

本目录中的 `train.jsonl` 和 `test.jsonl` 是针对当前 FAISS 知识库（`faiss/chunks.json`）人工整理的检索增强问答数据。

## 数据规模

| 文件 | 样本数 | GPT-1 | GPT-2 | GPT-3 |
| --- | ---: | ---: | ---: | ---: |
| `train.jsonl` | 45 | 15 | 15 | 15 |
| `test.jsonl` | 15 | 5 | 5 | 5 |

问题和答案使用英文，因为当前知识库论文正文为英文，并且 `BM25.py` 的分词器只提取英文单词和数字。这样可同时用于 BM25、向量检索和重排序器。

## 字段说明

每行都是一个独立 JSON 对象：

- `id`：样本的唯一标识。
- `question`：用户查询。
- `answer`：由原论文内容支持的参考答案。
- `source`：答案来源 PDF。
- `pages`：相关分块覆盖的 PDF 页码。
- `relevant_chunk_ids`：一个或多个正确分块，对应 `faiss/chunks.json` 中的 `metadata.chunk_id`。
- `hard_negative_chunk_ids`：主题相近但不能正确回答问题的分块，可用于训练 CrossEncoder 重排序器。
- `category`：问题类别。
- `difficulty`：人工标记的难度，取值为 `easy`、`medium` 或 `hard`。

## 推荐用途

检索训练时，可将 `(question, relevant chunk)` 作为正样本，将 `(question, hard negative chunk)` 作为负样本。测试时，建议至少统计：

- `Recall@K`：前 K 个检索结果是否包含任意 `relevant_chunk_ids`。
- `MRR@K`：第一个正确分块排名的倒数。
- `Exact Match / Token F1`：生成答案与 `answer` 的匹配程度。

为了避免答案泄漏，请勿将 `test.jsonl` 用于训练或超参数选择。
