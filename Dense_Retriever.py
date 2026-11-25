import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import json
import faiss
from pathlib import Path

#tokenizer和chunk 得到token_ids
#token_ids 怎么得到embeddings
class BGEEmbedding:

    def __init__(
        self,
        tokenizer,
        model,
        batch_size=4,
        max_length=1024,
        device=None
    ):
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size 和 max_length 必须大于 0")

        # 1. 自动选择设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length

        # 2. 加载 Tokenizer
        self.tokenizer = tokenizer

        # 3. 加载预训练 Transformer Encoder
        self.model = model

        self.model.to(self.device)
        self.model.eval()

        # 4. 获取模型隐藏层维度
        self.embedding_dim = self.model.config.hidden_size

        print(f"Embedding device: {self.device}")
        print(f"Embedding dimension: {self.embedding_dim}")

    def embed_texts(self, texts):
        """
        输入:
            str 或 list[str]

        输出:
            np.ndarray, shape [N, 1024]
        """

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty(
                (0, self.embedding_dim),
                dtype=np.float32
            )

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError("输入文本不能包含空字符串")

        embeddings = []

        # 5. 批量处理文本
        for start in range(0, len(texts), self.batch_size):

            batch_texts = texts[
                start:start + self.batch_size
            ]

            # 6. Tokenization
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=False,
                return_tensors="pt"
            )

            # 避免静默截断文档
            if inputs["input_ids"].shape[1] > self.max_length:
                raise ValueError(
                    "输入超过 max_length，请减小 Chunk Size"
                )

            # 将输入移动到 GPU 或 CPU
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            # 7. Transformer 前向传播
            with torch.inference_mode():

                outputs = self.model(**inputs)

                # 8. CLS Pooling
                sentence_embeddings = (
                    outputs.last_hidden_state[:, 0, :]
                )

                # 9. L2 归一化
                sentence_embeddings = F.normalize(
                    sentence_embeddings.float(),
                    p=2,
                    dim=-1
                )

            # 10. 移动到 CPU，转换为 NumPy
            vectors = sentence_embeddings.cpu().numpy()

            embeddings.append(vectors)

        # 11. 合并各 Batch 的结果
        return np.concatenate(embeddings, axis=0)

    def embed_chunks(self, chunks):
        """将 Chunker 输出的所有 Chunk 转换成向量。"""

        texts = [chunk["text"] for chunk in chunks] 

        return self.embed_texts(texts)



save_dir=Path("data/faiss")
save_dir.mkdir(parents=True,exist_ok=True)

class FAISSStore:
    def __init__(self,dim=1024):
        self.dim=dim
        self.index=faiss.IndexFlatIP(dim)
        self.chunks=[]

    def add(self,embeddings,chunks):
        vectors = np.ascontiguousarray(
            embeddings, dtype=np.float32
        )
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError("向量维度不正确")

        if len(vectors) != len(chunks):
            raise ValueError("向量数量与 Chunk 数量不一致")

        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.chunks.extend(chunks)
        
    def save(self,path):
        path=Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index,str(path/"index.faiss"))
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f,ensure_ascii=False, indent=2)

        pass

    def load(self,path):
        path = Path(path)


        self.index = faiss.read_index(
            str(path / "index.faiss")
        )

        with open(
            path / "chunks.json", "r", encoding="utf-8"
        ) as f:
            self.chunks = json.load(f)

        self.validate_chunks(self.chunks)
        
        if self.index.d != self.dim:
            raise ValueError("索引维度不匹配")

        if self.index.ntotal != len(self.chunks):
            raise ValueError("索引与 Chunk 数量不匹配")
        
    def search(self,query_embedding,top_k=20):
        query = np.asarray(
            query_embedding, dtype=np.float32
        ).reshape(1, -1)

        if query.shape[1] != self.dim:
            raise ValueError("查询向量维度不正确")

        query = np.ascontiguousarray(query)
        faiss.normalize_L2(query)

        k = min(top_k, self.index.ntotal)

        if k==0:
            return []

        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    "score": float(score),
                    "chunk": self.chunks[int(idx)],
                    "retriever":["FAISS"]
                })

        return results

    @staticmethod
    def validate_chunks(chunks, existing_ids=None):
        if not isinstance(chunks, list):
            raise TypeError("chunks 必须是列表")

        existing_ids = existing_ids or set()
        batch_ids = set()

        for position, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise TypeError(
                    f"第 {position} 个 chunk 不是字典"
                )

            text = chunk.get("text")
            metadata = chunk.get("metadata")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"第 {position} 个 chunk 缺少有效 text"
                )

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"第 {position} 个 chunk 缺少 metadata"
                )

            chunk_id = metadata.get("chunk_id")

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    f"第 {position} 个 chunk 缺少 chunk_id"
                )

            if chunk_id in batch_ids:
                raise ValueError(
                    f"本批数据存在重复 chunk_id: {chunk_id}"
                )

            if chunk_id in existing_ids:
                raise ValueError(
                    f"chunk_id 已经存在于索引中: {chunk_id}"
                )

            batch_ids.add(chunk_id)

        return batch_ids
        


if __name__ == "__main__":
    from pathlib import Path

    from chunker import Chunker
    from text_loader.pdf_loader import load_pdf

    # 使用相对路径，避免代码只能在某一台电脑上运行。
    project_dir = Path(__file__).resolve().parent
    pdf_path = project_dir / "data" / "raw" / "GPT-1.pdf"
    query = "How many Transformer layers does GPT-1 use?"
    top_k = 3

    if not pdf_path.is_file():
        raise FileNotFoundError(f"测试文档不存在: {pdf_path}")

    # 1. 初始化模型，并复用同一个 tokenizer 进行分块。
    embedder = BGEEmbedding()
    chunker = Chunker(
        tokenizer=embedder.tokenizer,
        chunk_size=512,
        chunk_overlap=64,
    )

    # 2. 加载文档并生成 Chunk。
    documents = load_pdf(pdf_path)
    chunks = chunker.chunk_documents(documents)
    if not chunks:
        raise RuntimeError("未从测试文档中生成任何 Chunk")

    print(f"文档页数: {len(documents)}")
    print(f"Chunk 数量: {len(chunks)}")

    # 3. 生成文档向量和查询向量。
    embeddings = embedder.embed_chunks(chunks)
    query_vector = embedder.encode(query)

    # 4. 检查输出的形状、数值和 L2 归一化结果。
    expected_shape = (len(chunks), embedder.embedding_dim)
    if embeddings.shape != expected_shape:
        raise AssertionError(
            f"Chunk 向量形状错误: {embeddings.shape}, "
            f"期望: {expected_shape}"
        )
    if query_vector.shape != (1, embedder.embedding_dim):
        raise AssertionError(f"查询向量形状错误: {query_vector.shape}")
    if not np.isfinite(embeddings).all() or not np.isfinite(query_vector).all():
        raise AssertionError("向量中包含 NaN 或 Inf")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise AssertionError("Chunk 向量未正确进行 L2 归一化")

    print(f"Chunk 向量形状: {embeddings.shape}")
    print(f"查询向量形状: {query_vector.shape}")

    # 5. 归一化向量的点积等价于余弦相似度。
    scores =   embeddings @ query_vector[0]
    result_count = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[-result_count:][::-1]

    print(f"\n查询: {query}")
    print(f"Top-{result_count} 检索结果:")

    # 6. 打印检索结果。
    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]
        metadata = chunk["metadata"]

        print(f"\n{'=' * 20} 第 {rank} 名 {'=' * 20}")
        print(f"相似度: {scores[index]:.6f}")
        print(f"来源: {metadata['source']}")
        print(f"页码: {metadata['pages']}")
        print(f"Chunk ID: {metadata['chunk_id']}")
        print("内容:")
        print(chunk["text"])
