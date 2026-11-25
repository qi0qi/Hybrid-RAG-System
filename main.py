from BM25 import BM25Retriever
from Dense_Retriever import FAISSStore,BGEEmbedding
from hybrid_retriever import hybrid_retirever
from text_loader.pdf_loader import load_pdf
from chunker import Chunker
from transformers import AutoTokenizer, AutoModel
from llm_generator import LLMGenerator
if __name__=="__main__":
    #1.添加流程
    tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-m3"
        )
    model=AutoModel.from_pretrained(
            "BAAI/bge-m3"
        )
    
    #path=r"data\raw\LLaMA.pdf"
    #docs=load_pdf(path)
    #chunker=Chunker(tokenizer)
    #chunks=chunker.chunk_documents(docs)

    faiss=FAISSStore()
    faiss_path=r"data\faiss"
    faiss.load(faiss_path)

    BGEembedding=BGEEmbedding(tokenizer,model)
    #embeddings=BGEembedding.embed_chunks(chunks)
    #faiss.add(embeddings,chunks)
    #faiss.save(r"data\faiss")

    BM_Retriever=BM25Retriever()
    BM_Retriever.load(r"data\faiss\chunks.json")
    
    #2.进行检索
    query=r"Why was BooksCorpus better suited to GPT-1 pre-training than the similarly sized 1B Word Benchmark?"
    #query_embeddings=BGEembedding.embed_texts(query)
    #results_faiss=faiss.search(query_embeddings,20)

    #results_BM25=BM_Retriever.search(query,20)

    hybridretirever=hybrid_retirever(BM_Retriever,faiss,BGEembedding)
    results=hybridretirever.search(query)


    generator = LLMGenerator()

    # 模拟 Reranker 返回的 Top-K Chunk

    answer = generator.generate(
        query="What is the purpose of causal masking?",
        results=results
    )

    print(answer["answer"])
    print(answer["sources"])


    