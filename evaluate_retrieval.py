"""Evaluate hybrid retrieval and reranking on the local QA test set.

Example:
    python evaluate_retrieval.py --k-values 1,3,5,10 --candidate-k 20

The script evaluates the hybrid ranking before reranking and the ranking after
CrossEncoder reranking on exactly the same candidate set. Model loading time is
excluded from latency measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load and validate the evaluation JSONL file."""
    samples: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是合法 JSON: {exc}"
                ) from exc

            sample_id = sample.get("id")
            question = sample.get("question")
            relevant_ids = sample.get("relevant_chunk_ids")

            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"{path} 第 {line_number} 行缺少有效 id")
            if sample_id in seen_sample_ids:
                raise ValueError(f"测试集存在重复样本 id: {sample_id}")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"样本 {sample_id} 缺少有效 question")
            if not isinstance(relevant_ids, list) or not relevant_ids:
                raise ValueError(
                    f"样本 {sample_id} 缺少 relevant_chunk_ids"
                )
            if any(
                not isinstance(chunk_id, str) or not chunk_id.strip()
                for chunk_id in relevant_ids
            ):
                raise ValueError(
                    f"样本 {sample_id} 包含无效 relevant_chunk_ids"
                )

            seen_sample_ids.add(sample_id)
            samples.append(sample)

            if limit is not None and len(samples) >= limit:
                break

    if not samples:
        raise ValueError(f"测试集为空: {path}")
    return samples


class evaluate_retrieval:
    """Calculate retrieval/reranking metrics and latency statistics.

    Each public ``*_at_k`` method evaluates one query. ``evaluate_rankings``
    macro-averages the per-query values over a complete test set.
    """

    METRIC_LABELS = (
        "Recall",
        "MRR",
        "NDCG",
        "Precision",
        "Hit Rate",
    )

    def __init__(self, k_values: Sequence[int] = (1, 3, 5, 10)) -> None:
        normalized_k_values = sorted({int(k) for k in k_values})
        if not normalized_k_values or any(k <= 0 for k in normalized_k_values):
            raise ValueError("k_values 必须包含至少一个正整数")
        self.k_values = tuple(normalized_k_values)

    @staticmethod
    def _unique_ranking(ranked_ids: Iterable[str]) -> list[str]:
        """Remove duplicate IDs while retaining the first (best) rank."""
        unique_ids: list[str] = []
        seen: set[str] = set()
        for chunk_id in ranked_ids:
            chunk_id = str(chunk_id)
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_ids.append(chunk_id)
        return unique_ids

    @classmethod
    def _prepare_at_k(
        cls,
        ranked_ids: Sequence[str],
        relevant_ids: Sequence[str] | set[str],
        k: int,
    ) -> tuple[list[str], set[str]]:
        if k <= 0:
            raise ValueError("k 必须大于 0")
        relevant = {str(chunk_id) for chunk_id in relevant_ids}
        if not relevant:
            raise ValueError("relevant_ids 不能为空")
        return cls._unique_ranking(ranked_ids)[:k], relevant

    # 1. Recall@K：前 K 项召回的相关分块占全部相关分块的比例。
    @classmethod
    def recall_at_k(
        cls,
        ranked_ids: Sequence[str],
        relevant_ids: Sequence[str] | set[str],
        k: int,
    ) -> float:
        top_k, relevant = cls._prepare_at_k(ranked_ids, relevant_ids, k)
        relevant_count = sum(chunk_id in relevant for chunk_id in top_k)
        return relevant_count / len(relevant)

    # 2. MRR@K：前 K 项中第一个相关分块排名的倒数。
    @classmethod
    def mrr_at_k(
        cls,
        ranked_ids: Sequence[str],
        relevant_ids: Sequence[str] | set[str],
        k: int,
    ) -> float:
        top_k, relevant = cls._prepare_at_k(ranked_ids, relevant_ids, k)
        for rank, chunk_id in enumerate(top_k, start=1):
            if chunk_id in relevant:
                return 1.0 / rank
        return 0.0

    # 3. NDCG@K：使用二值相关性衡量所有相关分块的排序质量。
    @classmethod
    def ndcg_at_k(
        cls,
        ranked_ids: Sequence[str],
        relevant_ids: Sequence[str] | set[str],
        k: int,
    ) -> float:
        top_k, relevant = cls._prepare_at_k(ranked_ids, relevant_ids, k)
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, chunk_id in enumerate(top_k, start=1)
            if chunk_id in relevant
        )
        ideal_relevant_count = min(len(relevant), k)
        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_relevant_count + 1)
        )
        return dcg / idcg if idcg else 0.0

    # 4. Precision@K：前 K 个位置中相关分块所占比例。
    @classmethod
    def precision_at_k(
        cls,
        ranked_ids: Sequence[str],
        relevant_ids: Sequence[str] | set[str],
        k: int,
    ) -> float:
        top_k, relevant = cls._prepare_at_k(ranked_ids, relevant_ids, k)
        relevant_count = sum(chunk_id in relevant for chunk_id in top_k)
        # 按标准定义除以 K；不足 K 个结果的位置视为不相关。
        return relevant_count / k

    # 5. Hit Rate@K：前 K 项中只要存在相关分块就记为 1，否则为 0。
    @classmethod
    def hit_rate_at_k(
        cls,
        ranked_ids: Sequence[str],
        relevant_ids: Sequence[str] | set[str],
        k: int,
    ) -> float:
        top_k, relevant = cls._prepare_at_k(ranked_ids, relevant_ids, k)
        return float(any(chunk_id in relevant for chunk_id in top_k))

    def evaluate_query(
        self,
        ranked_ids: Sequence[str],
        relevant_ids: Sequence[str] | set[str],
    ) -> dict[str, float]:
        """Calculate every configured metric for one query."""
        metrics: dict[str, float] = {}
        for k in self.k_values:
            metrics[f"Recall@{k}"] = self.recall_at_k(
                ranked_ids, relevant_ids, k
            )
            metrics[f"MRR@{k}"] = self.mrr_at_k(
                ranked_ids, relevant_ids, k
            )
            metrics[f"NDCG@{k}"] = self.ndcg_at_k(
                ranked_ids, relevant_ids, k
            )
            metrics[f"Precision@{k}"] = self.precision_at_k(
                ranked_ids, relevant_ids, k
            )
            metrics[f"Hit Rate@{k}"] = self.hit_rate_at_k(
                ranked_ids, relevant_ids, k
            )
        return metrics

    def evaluate_rankings(
        self,
        rankings: Sequence[Sequence[str]],
        relevant_id_sets: Sequence[Sequence[str]],
    ) -> dict[str, float]:
        """Macro-average all metrics over a complete evaluation set."""
        if len(rankings) != len(relevant_id_sets):
            raise ValueError("rankings 与 relevant_id_sets 数量不一致")
        if not rankings:
            raise ValueError("没有可评估的排名结果")

        totals = {
            f"{label}@{k}": 0.0
            for k in self.k_values
            for label in self.METRIC_LABELS
        }
        for ranked_ids, relevant_ids in zip(rankings, relevant_id_sets):
            for metric_name, value in self.evaluate_query(
                ranked_ids, relevant_ids
            ).items():
                totals[metric_name] += value

        query_count = len(rankings)
        return {
            metric_name: total / query_count
            for metric_name, total in totals.items()
        }

    @staticmethod
    def _percentile(values: Sequence[float], percent: float) -> float:
        if not values:
            return 0.0
        if not 0 <= percent <= 100:
            raise ValueError("percent 必须在 0 到 100 之间")

        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])

        position = (len(ordered) - 1) * percent / 100
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])

        weight = position - lower
        return float(
            ordered[lower] * (1 - weight)
            + ordered[upper] * weight
        )

    # 6. Latency：汇总每个查询耗时的均值、P50、P95、最小值和最大值。
    @classmethod
    def latency(
        cls,
        latencies_ms: Sequence[float],
    ) -> dict[str, float | int]:
        if any(latency < 0 for latency in latencies_ms):
            raise ValueError("latency 不能为负数")
        if not latencies_ms:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }

        return {
            "count": len(latencies_ms),
            "mean_ms": float(statistics.fmean(latencies_ms)),
            "p50_ms": cls._percentile(latencies_ms, 50),
            "p95_ms": cls._percentile(latencies_ms, 95),
            "min_ms": float(min(latencies_ms)),
            "max_ms": float(max(latencies_ms)),
        }

    @classmethod
    def extract_result_ids(
        cls,
        results: Sequence[dict[str, Any]],
    ) -> list[str]:
        """Extract unique chunk IDs from retrieval results."""
        chunk_ids: list[str] = []
        for position, result in enumerate(results, start=1):
            chunk_id = result.get("id")
            if chunk_id is None:
                chunk_id = result.get("metadata", {}).get("chunk_id")
            if chunk_id is None:
                chunk_id = (
                    result.get("chunk", {})
                    .get("metadata", {})
                    .get("chunk_id")
                )
            if chunk_id is None:
                raise KeyError(f"第 {position} 个检索结果没有 chunk_id")
            chunk_ids.append(str(chunk_id))
        return cls._unique_ranking(chunk_ids)


def synchronize_cuda(enabled: bool) -> None:
    """Synchronize CUDA so GPU timings include completed kernels."""
    if not enabled:
        return
    import torch

    torch.cuda.synchronize()


def build_components(args: argparse.Namespace) -> tuple[Any, Any, Any, bool]:
    """Load stores and models required by the evaluation pipeline."""
    from transformers import AutoModel, AutoTokenizer

    from BM25 import BM25Retriever
    from Dense_Retriever import BGEEmbedding, FAISSStore
    from Reranker import Reranker
    from hybrid_retriever import hybrid_retirever

    print(f"加载嵌入模型: {args.embedding_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    model = AutoModel.from_pretrained(args.embedding_model)
    embedding = BGEEmbedding(
        tokenizer,
        model,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        device=args.device,
    )

    faiss_store = FAISSStore(dim=embedding.embedding_dim)
    faiss_store.load(args.faiss_dir)

    bm25 = BM25Retriever().load(args.faiss_dir / "chunks.json")
    hybrid = hybrid_retirever(
        bm25,
        faiss_store,
        embedding,
        rrf_k=args.rrf_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
    )

    print(f"加载重排序模型: {args.reranker_model}")
    reranker = Reranker(
        model_name=args.reranker_model,
        device=args.device,
        max_length=args.reranker_max_length,
    )

    using_cuda = str(embedding.device).startswith("cuda")
    return hybrid, reranker, embedding, using_cuda


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Run retrieval and reranking evaluation."""
    samples = load_jsonl(args.test_file, limit=args.limit)
    evaluator = evaluate_retrieval(args.k_values)
    max_k = max(args.k_values)
    if args.candidate_k < max_k:
        raise ValueError(
            f"candidate_k={args.candidate_k} 不能小于最大评估 K={max_k}"
        )

    hybrid, reranker, _embedding, using_cuda = build_components(args)

    warmup_count = min(args.warmup_queries, len(samples))
    if warmup_count:
        print(f"预热 {warmup_count} 个查询（不计入延迟）")
        for sample in samples[:warmup_count]:
            candidates = hybrid.search(
                sample["question"],
                top_k=args.candidate_k,
                candidate_k=args.candidate_k,
            )
            reranker.rerank(
                sample["question"],
                candidates,
                top_k=args.candidate_k,
                batch_size=args.reranker_batch_size,
            )
        synchronize_cuda(using_cuda)

    hybrid_rankings: list[list[str]] = []
    reranked_rankings: list[list[str]] = []
    relevant_id_sets: list[list[str]] = []
    retrieval_latencies_ms: list[float] = []
    reranking_latencies_ms: list[float] = []
    end_to_end_latencies_ms: list[float] = []
    query_details: list[dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        question = sample["question"]
        relevant_ids = sample["relevant_chunk_ids"]

        synchronize_cuda(using_cuda)
        start = time.perf_counter()
        candidates = hybrid.search(
            question,
            top_k=args.candidate_k,
            candidate_k=args.candidate_k,
        )
        synchronize_cuda(using_cuda)
        retrieval_finished = time.perf_counter()

        reranked = reranker.rerank(
            question,
            candidates,
            top_k=args.candidate_k,
            batch_size=args.reranker_batch_size,
        )
        synchronize_cuda(using_cuda)
        reranking_finished = time.perf_counter()

        hybrid_ids = evaluator.extract_result_ids(candidates)
        reranked_ids = evaluator.extract_result_ids(reranked)
        retrieval_ms = (retrieval_finished - start) * 1000
        reranking_ms = (reranking_finished - retrieval_finished) * 1000
        end_to_end_ms = (reranking_finished - start) * 1000

        hybrid_rankings.append(hybrid_ids)
        reranked_rankings.append(reranked_ids)
        relevant_id_sets.append(list(relevant_ids))
        retrieval_latencies_ms.append(retrieval_ms)
        reranking_latencies_ms.append(reranking_ms)
        end_to_end_latencies_ms.append(end_to_end_ms)

        query_details.append(
            {
                "id": sample["id"],
                "question": question,
                "relevant_chunk_ids": relevant_ids,
                "hybrid_ranked_ids": hybrid_ids,
                "reranked_ids": reranked_ids,
                "metrics": {
                    "hybrid": evaluator.evaluate_query(
                        hybrid_ids, relevant_ids
                    ),
                    "hybrid_reranked": evaluator.evaluate_query(
                        reranked_ids, relevant_ids
                    ),
                },
                "latency_ms": {
                    "hybrid_retrieval": retrieval_ms,
                    "reranking": reranking_ms,
                    "end_to_end": end_to_end_ms,
                },
            }
        )
        print(f"[{index:>3}/{len(samples)}] {sample['id']}")

    hybrid_metrics = evaluator.evaluate_rankings(
        hybrid_rankings,
        relevant_id_sets,
    )
    reranked_metrics = evaluator.evaluate_rankings(
        reranked_rankings,
        relevant_id_sets,
    )

    return {
        "config": {
            "test_file": str(args.test_file),
            "faiss_dir": str(args.faiss_dir),
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model,
            "candidate_k": args.candidate_k,
            "k_values": list(args.k_values),
            "query_count": len(samples),
            "warmup_queries": warmup_count,
            "device": args.device,
        },
        "metrics": {
            "hybrid": hybrid_metrics,
            "hybrid_reranked": reranked_metrics,
        },
        "latency": {
            "hybrid_retrieval": evaluator.latency(retrieval_latencies_ms),
            "reranking": evaluator.latency(reranking_latencies_ms),
            "end_to_end": evaluator.latency(end_to_end_latencies_ms),
        },
        "queries": query_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估混合检索及 Reranker 的排序质量与延迟"
    )
    parser.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/faiss"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/retrieval_evaluation_results.json"),
    )
    parser.add_argument("--k-values", default="1,3,5,10")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup-queries", type=int, default=1)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    args = parser.parse_args()

    try:
        args.k_values = tuple(
            sorted({int(value.strip()) for value in args.k_values.split(",")})
        )
    except ValueError:
        parser.error("--k-values 必须是逗号分隔的整数，例如 1,3,5,10")
    if not args.k_values or any(k <= 0 for k in args.k_values):
        parser.error("所有 K 值都必须大于 0")
    if args.candidate_k <= 0:
        parser.error("--candidate-k 必须大于 0")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须大于 0")
    if args.warmup_queries < 0:
        parser.error("--warmup-queries 不能小于 0")
    if args.embedding_batch_size <= 0 or args.reranker_batch_size <= 0:
        parser.error("batch size 必须大于 0")
    if args.rrf_k < 0:
        parser.error("--rrf-k 不能小于 0")

    results = run_evaluation(args)
    print("\n=== Ranking metrics (macro average) ===")
    for stage, metrics in results["metrics"].items():
        print(f"\n{stage}")
        for k in args.k_values:
            values = "  ".join(
                f"{name}={metrics[f'{name}@{k}']:.4f}"
                for name in evaluate_retrieval.METRIC_LABELS
            )
            print(f"K={k}: {values}")

    print("\n=== Latency per query (ms) ===")
    for stage, summary in results["latency"].items():
        print(
            f"{stage}: mean={summary['mean_ms']:.2f}, "
            f"p50={summary['p50_ms']:.2f}, p95={summary['p95_ms']:.2f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
