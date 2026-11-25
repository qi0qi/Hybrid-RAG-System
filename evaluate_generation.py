"""Evaluate generated RAG answers with RAGAS generation-quality metrics.

Metrics:
    - Faithfulness: whether response claims are supported by retrieved contexts.
    - Answer Correctness: factual and semantic agreement with the reference answer.
    - Answer Relevancy: how directly the response addresses the question.

Example:
    python evaluate_generation.py --limit 5

The answer generator and metric judge both use the local Ollama OpenAI-compatible
endpoint by default. For a more reliable evaluation, use a judge model that is
at least as capable as, and preferably different from, the generation model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from evaluate_retrieval import build_components, load_jsonl, synchronize_cuda


class evaluate_generation:
    """Calculate RAG generation metrics and aggregate evaluation results."""

    METRIC_NAMES = (
        "Faithfulness",
        "Answer Correctness",
        "Answer Relevancy",
    )

    def __init__(self, scorers: dict[str, Any]) -> None:
        missing = set(self.METRIC_NAMES) - scorers.keys()
        if missing:
            raise ValueError(f"缺少生成评价器: {', '.join(sorted(missing))}")
        self.scorers = scorers

    @classmethod
    def from_models(
        cls,
        args: argparse.Namespace,
        project_embedding: Any,
    ) -> "evaluate_generation":
        """Create RAGAS scorers while reusing the project's BGE embeddings."""
        try:
            from openai import OpenAI
            from ragas.embeddings.base import BaseRagasEmbedding
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerCorrectness,
                AnswerRelevancy,
                Faithfulness,
            )
        except ImportError as exc:
            raise ImportError(
                "生成质量评估需要 RAGAS collections API。请安装："
                "pip install 'ragas>=0.4' instructor openai"
            ) from exc

        class ProjectBGEEmbeddings(BaseRagasEmbedding):
            def __init__(self, embedding_model: Any) -> None:
                super().__init__()
                self.embedding_model = embedding_model

            def embed_text(self, text: str, **kwargs: Any) -> list[float]:
                vector = self.embedding_model.embed_texts(text)[0]
                return vector.astype(float).tolist()

            async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
                return await asyncio.to_thread(self.embed_text, text, **kwargs)

        client = OpenAI(
            api_key=args.judge_api_key,
            base_url=args.judge_base_url,
            timeout=args.judge_timeout,
            max_retries=args.judge_max_retries,
        )
        judge_llm = llm_factory(
            args.judge_model,
            provider="openai",
            client=client,
            temperature=0.0,
        )
        judge_embeddings = ProjectBGEEmbeddings(project_embedding)
        return cls({
            "Faithfulness": Faithfulness(llm=judge_llm),
            "Answer Correctness": AnswerCorrectness(
                llm=judge_llm,
                embeddings=judge_embeddings,
            ),
            "Answer Relevancy": AnswerRelevancy(
                llm=judge_llm,
                embeddings=judge_embeddings,
            ),
        })

    @staticmethod
    def _serialize(result: Any) -> dict[str, Any]:
        value = float(result.value)
        if not math.isfinite(value):
            raise ValueError(f"指标返回了非有限数值: {value}")
        reason = getattr(result, "reason", None)
        if reason is not None and not isinstance(reason, (str, int, float, bool)):
            reason = str(reason)
        return {"score": value, "reason": reason}
    #1.faithfulness
    def faithfulness(
        self,
        question: str,
        response: str,
        contexts: Sequence[str],
    ) -> dict[str, Any]:
        """Evaluate whether the response is supported by retrieved contexts."""
        if not contexts:
            return {
                "score": 0.0,
                "reason": "没有检索上下文，回答无法被上下文支持。",
            }
        return self._serialize(self.scorers["Faithfulness"].score(
            user_input=question,
            response=response,
            retrieved_contexts=list(contexts),
        ))
    #2.answer_correctness
    def answer_correctness(
        self,
        question: str,
        response: str,
        reference: str,
    ) -> dict[str, Any]:
        """Evaluate factual and semantic agreement with the reference answer."""
        return self._serialize(self.scorers["Answer Correctness"].score(
            user_input=question,
            response=response,
            reference=reference,
        ))
    #3.answer_relevancy
    def answer_relevancy(
        self,
        question: str,
        response: str,
    ) -> dict[str, Any]:
        """Evaluate how directly the response addresses the question."""
        return self._serialize(self.scorers["Answer Relevancy"].score(
            user_input=question,
            response=response,
        ))

    def evaluate_query(
        self,
        *,
        question: str,
        response: str,
        reference: str,
        contexts: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """Calculate every generation metric for one answer."""
        return {
            "Faithfulness": self.faithfulness(question, response, contexts),
            "Answer Correctness": self.answer_correctness(
                question, response, reference
            ),
            "Answer Relevancy": self.answer_relevancy(question, response),
        }

    @classmethod
    def aggregate(
        cls,
        query_details: Sequence[dict[str, Any]],
    ) -> dict[str, float]:
        """Macro-average all metrics over the evaluation set."""
        if not query_details:
            raise ValueError("没有可聚合的生成评估结果")
        return {
            name: float(statistics.fmean(
                detail["metrics"][name]["score"] for detail in query_details
            ))
            for name in cls.METRIC_NAMES
        }

    @staticmethod
    def latency(values: Sequence[float]) -> dict[str, float | int]:
        """Summarize latency values."""
        if not values:
            return {"count": 0, "mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        return {
            "count": len(values),
            "mean_ms": float(statistics.fmean(values)),
            "min_ms": float(min(values)),
            "max_ms": float(max(values)),
        }

    @staticmethod
    def result_text(result: dict[str, Any]) -> str:
        chunk = result.get("chunk")
        if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
            return chunk["text"]
        if isinstance(result.get("text"), str):
            return result["text"]
        raise ValueError("检索结果缺少有效文本")

    @staticmethod
    def result_chunk_id(result: dict[str, Any]) -> str:
        chunk = result.get("chunk", {})
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        chunk_id = metadata.get("chunk_id")
        if chunk_id is None:
            raise KeyError("检索结果缺少 chunk.metadata.chunk_id")
        return str(chunk_id)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Run retrieval, reranking, answer generation, and generation evaluation."""
    try:
        from llm_generator import LLMGenerator
    except ImportError as exc:
        raise ImportError(
            "回答生成需要 OpenAI 兼容客户端。请安装：pip install openai"
        ) from exc

    samples = load_jsonl(args.test_file, limit=args.limit)

    hybrid, reranker, embedding, using_cuda = build_components(args)
    generator = LLMGenerator(model_name=args.generator_model)
    evaluator = evaluate_generation.from_models(args, embedding)

    query_details = []
    generation_latencies_ms = []
    evaluation_latencies_ms = []
    end_to_end_latencies_ms = []

    for index, sample in enumerate(samples, start=1):
        question = sample["question"]
        reference = sample["answer"]

        synchronize_cuda(using_cuda)
        start = time.perf_counter()
        candidates = hybrid.search(
            question,
            top_k=args.candidate_k,
            candidate_k=args.candidate_k,
        )
        reranked = reranker.rerank(
            question,
            candidates,
            top_k=args.context_k,
            batch_size=args.reranker_batch_size,
        )
        generation = generator.generate(
            query=question,
            results=reranked,
        )
        synchronize_cuda(using_cuda)
        generation_finished = time.perf_counter()

        response = generation["answer"]
        contexts = [evaluator.result_text(result) for result in reranked]
        metric_details = evaluator.evaluate_query(
            question=question,
            response=response,
            reference=reference,
            contexts=contexts,
        )
        evaluation_finished = time.perf_counter()

        generation_ms = (generation_finished - start) * 1000
        evaluation_ms = (evaluation_finished - generation_finished) * 1000
        end_to_end_ms = (evaluation_finished - start) * 1000
        generation_latencies_ms.append(generation_ms)
        evaluation_latencies_ms.append(evaluation_ms)
        end_to_end_latencies_ms.append(end_to_end_ms)

        query_details.append({
            "id": sample["id"],
            "question": question,
            "reference_answer": reference,
            "generated_answer": response,
            "context_chunk_ids": [
                evaluator.result_chunk_id(result)
                for result in reranked
            ],
            "contexts": contexts,
            "metrics": metric_details,
            "latency_ms": {
                "retrieval_reranking_generation": generation_ms,
                "generation_evaluation": evaluation_ms,
                "end_to_end": end_to_end_ms,
            },
        })
        print(f"[{index:>3}/{len(samples)}] {sample['id']}")

    return {
        "config": {
            "test_file": str(args.test_file),
            "sample_count": len(samples),
            "candidate_k": args.candidate_k,
            "context_k": args.context_k,
            "generator_model": args.generator_model,
            "judge_model": args.judge_model,
            "judge_base_url": args.judge_base_url,
            "embedding_model": args.embedding_model,
        },
        "metrics": evaluator.aggregate(query_details),
        "latency_ms": {
            "retrieval_reranking_generation": evaluator.latency(
                generation_latencies_ms
            ),
            "generation_evaluation": evaluator.latency(
                evaluation_latencies_ms
            ),
            "end_to_end": evaluator.latency(end_to_end_latencies_ms),
        },
        "queries": query_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估 RAG 生成回答的忠实度、正确性与相关性"
    )
    parser.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/faiss"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generation_evaluation_results.json"),
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
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
    parser.add_argument("--generator-model", default="qwen3:4b")
    parser.add_argument("--judge-model", default="qwen3:4b")
    parser.add_argument("--judge-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--judge-api-key", default="ollama")
    parser.add_argument("--judge-timeout", type=float, default=600.0)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    args = parser.parse_args()

    if args.candidate_k <= 0 or args.context_k <= 0:
        parser.error("candidate_k 和 context_k 必须大于 0")
    if args.context_k > args.candidate_k:
        parser.error("context_k 不能大于 candidate_k")
    if args.context_k > 5:
        parser.error(
            "当前 LLMGenerator 最多使用前 5 个上下文，因此 context_k 不能大于 5"
        )
    if args.limit is not None and args.limit <= 0:
        parser.error("limit 必须大于 0")
    if args.embedding_batch_size <= 0 or args.reranker_batch_size <= 0:
        parser.error("batch size 必须大于 0")
    if args.rrf_k < 0:
        parser.error("rrf_k 不能小于 0")
    if args.bm25_weight < 0 or args.dense_weight < 0:
        parser.error("检索权重不能小于 0")
    if args.bm25_weight == 0 and args.dense_weight == 0:
        parser.error("BM25 和 Dense 权重不能同时为 0")
    if args.judge_timeout <= 0 or args.judge_max_retries < 0:
        parser.error("Judge timeout 必须大于 0，重试次数不能小于 0")

    results = run_evaluation(args)
    print("\nGeneration evaluation")
    print("=" * 46)
    for name in evaluate_generation.METRIC_NAMES:
        print(f"{name:<24} {results['metrics'][name]:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
