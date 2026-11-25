"""使用 CrossEncoder 对检索结果进行重排序。

Reranker 不负责从整个知识库中找文档，而是接收 BM25/FAISS 已经召回的
少量候选文档，再逐个判断「query 和候选文本是否相关」。
"""

from typing import Any, Dict, List, Sequence, Union

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


Candidate = Union[str, Dict[str, Any]]


class Reranker:
    """基于 BGE CrossEncoder 的重排序器。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = None,
        max_length: int = 512,
    ):
        """加载重排序模型。

        Args:
            model_name: Hugging Face 上的 CrossEncoder 模型名称或本地路径。
            device: "cuda"、"cpu" 或 None。None 表示由框架自动选择。
            max_length: query 和文档拼接后的最大 token 数。
        """
        if CrossEncoder is None:
            raise ImportError(
                "缺少 sentence-transformers，请先执行："
                "pip install sentence-transformers"
            )

        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )

    @staticmethod
    def _get_text(candidate: Candidate) -> str:
        """从字符串或项目中的检索结果字典里取出文本。"""
        if isinstance(candidate, str):
            return candidate

        if not isinstance(candidate, dict):
            raise TypeError("candidate 必须是字符串或字典")

        # BM25 和混合检索结果使用顶层 text。
        if isinstance(candidate.get("text"), str):
            return candidate["text"]

        # 原始 FAISS 结果的文本位于 result["chunk"]["text"]。
        chunk = candidate.get("chunk")
        if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
            return chunk["text"]

        raise ValueError("candidate 中没有找到 text 字段")

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int = 5,
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """计算 query 与每个候选文本的相关性，并按分数降序排列。

        返回结果统一为字典，并增加 rerank_score 字段。原检索器给出的
        score 会保留下来，便于对比召回分数与重排序分数。
        """
        if not query.strip() or top_k <= 0 or not candidates:
            return []

        texts = [self._get_text(candidate) for candidate in candidates]

        # CrossEncoder 会同时阅读 query 和 document，比单独比较向量更精细。
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        results = []
        for candidate, text, score in zip(candidates, texts, scores):
            if isinstance(candidate, dict):
                # 复制字典，避免修改调用者传入的原始检索结果。
                result = dict(candidate)
                result.setdefault("text", text)
            else:
                result = {"text": text}

            result["score"] = float(score)
            results.append(result)

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )
        return results[: min(top_k, len(results))]


if __name__ == "__main__":
    # 简单学习示例：实际使用时，把 candidates 换成混合检索器的返回结果即可。
    query = "What is the difference between GPT-1 and GPT-2?"
    candidates = [
        {
            "id": "chunk-1",
            "text": "GPT-1 is a generative pretrained Transformer.",
            "score": 0.03,
        },
        {
            "id": "chunk-2",
            "text": "GPT-2 uses a larger model and much more training data than GPT-1.",
            "score": 0.02,
        },
        {
            "id": "chunk-3",
            "text": "BERT uses bidirectional self-attention.",
            "score": 0.04,
        },
    ]

    reranker = Reranker()
    reranked_results = reranker.rerank(query, candidates, top_k=2)

    for rank, result in enumerate(reranked_results, start=1):
        print(f"Rank {rank}")
        print(f"Rerank score: {result['rerank_score']:.4f}")
        print(f"Text: {result['text']}\n")
