from BM25 import BM25Retriever
from Dense_Retriever import FAISSStore

class hybrid_retirever:
    def __init__(self,BM25,faiss,embedding_model,rrf_k=60,bm25_weight=1.0,dense_weight=1.0):
        self.bm25=BM25
        self.faiss=faiss
        self.embedding_model=embedding_model
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def search(self,query, top_k=5, candidate_k=20):
        if top_k <= 0:
            return []
        
        candidate_k = max(top_k, candidate_k)

        # 1. BM25 检索
        bm25_results = self.bm25.search(
            query, top_k=candidate_k
        )

        # 2. FAISS 向量检索
        query_embedding = self.embedding_model.embed_texts(query)

        faiss_results = self.faiss.search(
            query_embedding, top_k=candidate_k
        )

        # 3. RRF 融合
        fused = {}

        # 3.1 处理 BM25
        for rank, result in enumerate(bm25_results, 1):

            chunk = result["chunk"]
            chunk_id = str(chunk["metadata"]["chunk_id"])

            fused[chunk_id] = {
                "score": self.bm25_weight /
                         (self.rrf_k + rank),
                "chunk": chunk,
                "retriever":result["retriever"] + ["hybrid_retriever"]
            }

        # 3.2 处理 FAISS
        for rank, result in enumerate(faiss_results, 1):

            chunk = result["chunk"]
            chunk_id = str(chunk["metadata"]["chunk_id"])

            if chunk_id not in fused:

                fused[chunk_id] = {
                    "score": 0.0,
                    "chunk": chunk,
                    "retriever":result["retriever"] + ["hybrid_retriever"]
                }

            fused[chunk_id]["score"] += (
                self.dense_weight /
                (self.rrf_k + rank)
            )

            for source in result["retriever"]:
                if source not in fused[chunk_id]["retriever"]:
                    fused[chunk_id]["retriever"].insert(-1, source)
        
        # 4. 根据 RRF 分数重新排序
        ranked_results = sorted(
            fused.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked_results[:top_k]
