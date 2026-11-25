import numpy as np
import re
from pathlib import Path
import json
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self):
        self.bm25=None
        self.tokenized_corpus=[]
        self.chunks=[]

    #1.英文分词
    @staticmethod
    def tokenize(text):
        text=text.lower()
        tokens = re.findall(
            r"[a-z]+(?:[-'][a-z0-9]+)*|\d+(?:\.\d+)*",
            text
        )

        return tokens
    #2.构建BM25索引
    def build_index(self,chunks):
        self.chunks = []
        self.tokenized_corpus = []
        seen_ids = set()

        for chunk in chunks:

            chunk_id = chunk["metadata"]["chunk_id"]
            text = chunk["text"]

            if chunk_id in seen_ids:
                raise ValueError("Duplicate chunk ID")

            seen_ids.add(chunk_id)

            tokens = self.tokenize(text)

            if not tokens:
                continue

            self.chunks.append(dict(chunk))
            self.tokenized_corpus.append(tokens)

        if not self.chunks:
            raise ValueError("No valid chunks")

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )
    #3.TOP-K 搜索
    def search(self, query, top_k=20):

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built"
            )

        if top_k <= 0:
            return []

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        # 仅保留与查询至少共享一个词的文档
        query_set = set(query_tokens)

        matched_indices = [
            i
            for i, tokens in enumerate(
                self.tokenized_corpus
            )
            if query_set.intersection(tokens)
        ]

        # 按 BM25 分数降序排列
        ranked_indices = sorted(
            matched_indices,
            key=lambda i: (-scores[i], i)
        )

        results = []

        for i in ranked_indices[:top_k]:

            chunk = self.chunks[i]

            results.append({
                "score": float(scores[i]),
                "chunk": chunk,
                "retriever":["BM25"]
            })

        return results
     # 4. 保存索引重建所需的数据
    # 4. 保存索引重建所需的数据
    def save(self, path):

        if self.bm25 is None:
            raise RuntimeError(
                "No index to save"
            )

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = {
            "chunks": self.chunks
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

    # 5. 加载并重建 BM25 索引
    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 同时兼容原始 chunks.json 和 BM25.save() 的输出
        chunks = data if isinstance(data, list) else data["chunks"]
        self.build_index(chunks)
        return self

