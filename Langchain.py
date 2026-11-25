from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough
)


from BM25 import BM25Retriever
from Dense_Retriever import FAISSStore,BGEEmbedding
from hybrid_retriever import hybrid_retirever
from text_loader.pdf_loader import load_pdf
from chunker import Chunker
from transformers import AutoTokenizer, AutoModel
from llm_generator import LLMGenerator
from pathlib import Path
import time

from evaluate_retrieval import (
    evaluate_retrieval,
    load_jsonl,
    synchronize_cuda,
)

# result 外层字段在转换为 Document 后没有固定的存放位置，因此使用一个
# 保留字段保存它们。这样 score、rerank_score、retriever 等信息可以无损还原。
_RESULT_ENVELOPE_KEY = "_rag_result"
_MIRRORED_RESULT_FIELDS = ("score", "rerank_score", "retriever")


def results_to_documents(results: list[dict]) -> list[Document]:
    """将项目中的检索结果转换为 LangChain ``Document``。

    支持两种常见输入：
    1. ``{"chunk": {"text": ..., "metadata": ...}, "score": ...}``
    2. ``{"text": ..., "metadata": ..., "score": ...}``

    result 的外层字段会保存在 ``metadata["_rag_result"]`` 中；常用的
    score、rerank_score 和 retriever 也会镜像到 metadata 顶层，方便链中读取。
    """
    documents = []

    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise TypeError(f"第 {index} 个 result 必须是字典")

        chunk = result.get("chunk")
        if chunk is None:
            text = result.get("text")
            chunk_metadata = result.get("metadata", {})
        else:
            if not isinstance(chunk, dict):
                raise TypeError(f"第 {index} 个 result 的 chunk 必须是字典")
            text = chunk.get("text")
            chunk_metadata = chunk.get("metadata", {})

        if not isinstance(text, str):
            raise ValueError(f"第 {index} 个 result 缺少有效的文本")
        if not isinstance(chunk_metadata, dict):
            raise TypeError(f"第 {index} 个 result 的 metadata 必须是字典")
        if _RESULT_ENVELOPE_KEY in chunk_metadata:
            raise ValueError(
                f"metadata 中的 {_RESULT_ENVELOPE_KEY!r} 是转换函数的保留字段"
            )

        document_metadata = dict(chunk_metadata)
        structural_fields = (
            {"text", "metadata"}
            if chunk is None
            else {"chunk"}
        )
        result_fields = {
            key: value
            for key, value in result.items()
            if key not in structural_fields
        }

        # 常用检索字段放到 metadata 顶层，LangChain 链中可以直接访问。
        # mirrored_fields 用于反向转换时只移除本函数添加的字段。
        mirrored_fields = []
        for field in _MIRRORED_RESULT_FIELDS:
            if field in result_fields and field not in document_metadata:
                document_metadata[field] = result_fields[field]
                mirrored_fields.append(field)

        document_metadata[_RESULT_ENVELOPE_KEY] = {
            "fields": result_fields,
            "mirrored_fields": mirrored_fields,
        }

        documents.append(
            Document(
                page_content=text,
                metadata=document_metadata,
            )
        )

    return documents


def documents_to_results(documents: list[Document]) -> list[dict]:
    """将 LangChain ``Document`` 还原为项目使用的检索结果结构。"""
    results = []

    for index, document in enumerate(documents):
        if not isinstance(document, Document):
            raise TypeError(f"第 {index} 个元素必须是 LangChain Document")

        chunk_metadata = dict(document.metadata)
        envelope = chunk_metadata.pop(_RESULT_ENVELOPE_KEY, None)
        result_fields = {}

        if envelope is not None:
            if not isinstance(envelope, dict):
                raise TypeError(
                    f"第 {index} 个 Document 的 {_RESULT_ENVELOPE_KEY} 格式不正确"
                )

            fields = envelope.get("fields", {})
            mirrored_fields = envelope.get("mirrored_fields", [])

            if not isinstance(fields, dict) or not isinstance(mirrored_fields, list):
                raise TypeError(
                    f"第 {index} 个 Document 的 {_RESULT_ENVELOPE_KEY} 格式不正确"
                )

            result_fields.update(fields)
            for field in mirrored_fields:
                chunk_metadata.pop(field, None)

        result_fields["chunk"] = {
            "text": document.page_content,
            "metadata": chunk_metadata,
        }
        results.append(result_fields)

    return results

tokenizer = AutoTokenizer.from_pretrained(
        "BAAI/bge-m3"
    )
model=AutoModel.from_pretrained(
        "BAAI/bge-m3"
    )

faiss=FAISSStore()
faiss_path=r"data\faiss"
faiss.load(faiss_path)

BGEembedding=BGEEmbedding(tokenizer,model)

BM_Retriever=BM25Retriever()
BM_Retriever.load(r"data\faiss\chunks.json")
hybridretirever=hybrid_retirever(BM_Retriever,faiss,BGEembedding)


def retriever_documents(query:str)->list[Document]:
        
    ranked=hybridretirever.search(query)

    return results_to_documents(ranked)

retriever = RunnableLambda(retriever_documents)


def retriever_documents_for_evaluation(inputs: dict) -> list[Document]:
    """使用可配置的候选数量执行混合检索。"""
    if not isinstance(inputs, dict):
        raise TypeError("inputs 必须是字典")

    query = inputs.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("inputs['query'] 必须是非空字符串")

    top_k = int(inputs.get("top_k", 20))
    candidate_k = int(inputs.get("candidate_k", top_k))
    if top_k <= 0 or candidate_k <= 0:
        raise ValueError("top_k 和 candidate_k 必须大于 0")

    results = hybridretirever.search(
        query,
        top_k=top_k,
        candidate_k=candidate_k,
    )
    return results_to_documents(results)


evaluation_retriever_runnable = RunnableLambda(
    retriever_documents_for_evaluation
).with_config({
    "run_name": "hybrid_retriever_for_evaluation",
})


def evaluate_test_set(inputs: dict | None = None) -> dict:
    """通过 LangChain Runnable 在 JSONL 测试集上评估混合检索器。

    可配置字段：test_file、k_values、candidate_k、limit、warmup_queries。
    """
    if inputs is None:
        inputs = {}
    if not isinstance(inputs, dict):
        raise TypeError("inputs 必须是字典或 None")

    test_file = Path(inputs.get("test_file", "data/test.jsonl"))
    raw_k_values = inputs.get("k_values", (1, 3, 5, 10))
    if isinstance(raw_k_values, str):
        raw_k_values = [
            value.strip()
            for value in raw_k_values.split(",")
            if value.strip()
        ]

    evaluator = evaluate_retrieval(raw_k_values)
    candidate_k = int(inputs.get("candidate_k", 20))
    limit = inputs.get("limit")
    warmup_queries = int(inputs.get("warmup_queries", 1))

    if candidate_k <= 0:
        raise ValueError("candidate_k 必须大于 0")
    if candidate_k < max(evaluator.k_values):
        raise ValueError(
            f"candidate_k={candidate_k} 不能小于最大评估 K="
            f"{max(evaluator.k_values)}"
        )
    if limit is not None:
        limit = int(limit)
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
    if warmup_queries < 0:
        raise ValueError("warmup_queries 不能小于 0")

    samples = load_jsonl(test_file, limit=limit)
    using_cuda = str(BGEembedding.device).startswith("cuda")
    warmup_count = min(warmup_queries, len(samples))

    for sample in samples[:warmup_count]:
        evaluation_retriever_runnable.invoke({
            "query": sample["question"],
            "top_k": candidate_k,
            "candidate_k": candidate_k,
        })
    synchronize_cuda(using_cuda)

    rankings = []
    relevant_id_sets = []
    latencies_ms = []
    query_details = []

    for sample in samples:
        synchronize_cuda(using_cuda)
        start = time.perf_counter()
        documents = evaluation_retriever_runnable.invoke({
            "query": sample["question"],
            "top_k": candidate_k,
            "candidate_k": candidate_k,
        })
        synchronize_cuda(using_cuda)
        latency_ms = (time.perf_counter() - start) * 1000

        results = documents_to_results(documents)
        ranked_ids = evaluator.extract_result_ids(results)
        relevant_ids = list(sample["relevant_chunk_ids"])

        rankings.append(ranked_ids)
        relevant_id_sets.append(relevant_ids)
        latencies_ms.append(latency_ms)
        query_details.append({
            "id": sample["id"],
            "question": sample["question"],
            "relevant_chunk_ids": relevant_ids,
            "ranked_ids": ranked_ids,
            "metrics": evaluator.evaluate_query(ranked_ids, relevant_ids),
            "latency_ms": latency_ms,
        })

    return {
        "config": {
            "test_file": str(test_file),
            "k_values": list(evaluator.k_values),
            "candidate_k": candidate_k,
            "limit": limit,
            "warmup_queries": warmup_count,
        },
        "sample_count": len(samples),
        "metrics": evaluator.evaluate_rankings(rankings, relevant_id_sets),
        "latency_ms": evaluate_retrieval.latency(latencies_ms),
        "queries": query_details,
    }


evaluate_test_set_runnable = RunnableLambda(
    evaluate_test_set
).with_config({
    "run_name": "evaluate_hybrid_retriever_on_test_set",
})

generator = LLMGenerator()

def generate_answer(inputs: dict) -> dict:
    query = inputs["query"]
    documents = inputs["documents"]

    results = documents_to_results(documents)

    return generator.generate(
        query=query,
        results=results,
    )

INDEX_DIR = Path("data/faiss")
chunker = Chunker(tokenizer)


def add_chunks(inputs: dict) -> dict:
    """
    inputs:
        {"path": "data/raw/new.pdf"}
    """
    pdf_path = Path(inputs["path"])

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    # 1. 加载并切分 PDF
    docs = load_pdf(pdf_path)
    chunks = chunker.chunk_documents(docs)

    if not chunks:
        raise ValueError(f"PDF 中没有提取到有效内容：{pdf_path}")

    # 2. 防止同一篇 PDF 被重复添加
    existing_ids = {
        chunk["metadata"]["chunk_id"]
        for chunk in faiss.chunks
    }

    FAISSStore.validate_chunks(
        chunks,
        existing_ids=existing_ids,
    )

    # 3. 更新 FAISS 索引
    embeddings = BGEembedding.embed_chunks(chunks)
    faiss.add(embeddings, chunks)
    faiss.save(INDEX_DIR)

    # 4. chunks.json 已包含新旧所有 chunk，重新构建 BM25
    BM_Retriever.load(INDEX_DIR / "chunks.json")

    return {
        "status": "success",
        "file": pdf_path.name,
        "pages": len(docs),
        "chunks_added": len(chunks),
        "total_chunks": len(faiss.chunks),
    }

generator_runnable=RunnableLambda(generate_answer)

rag_chain = (
    {
        "query": RunnablePassthrough(),
        "documents": retriever,
    }
    | generator_runnable
)

add_document_chain = RunnableLambda(
    add_chunks
).with_config({
    "run_name": "add_pdf_to_knowledge_base"
})



if __name__ == "__main__":
    report = evaluate_test_set_runnable.invoke({
    "test_file": "data/test.jsonl",
    "k_values": [1, 3, 5, 10],
    "candidate_k": 20,
    "limit": None,
    "warmup_queries": 1,
    })

    print(report["metrics"])
    print(report["latency_ms"])
