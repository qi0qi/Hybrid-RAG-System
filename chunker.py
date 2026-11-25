from collections import defaultdict
from transformers import AutoTokenizer



class Chunker:
    def __init__(self,tokenizer,chunk_size=512,chunk_overlap=64):
        if chunk_size<=0:
            raise ValueError("chunk_size必须大于0")
        if chunk_overlap<0 or chunk_overlap>=chunk_size:
            raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")

        self.tokenizer=tokenizer
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap

    #文本转token_id
    def encoder(self,text):
        try:
            ids=self.tokenizer.encode(text,add_special_tokens=False)
        except TypeError:
            ids=self.tokenizer.encode(text)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()

        return list(ids)

    #token_id转文本
    def decoder(self,ids):
        try:
            return self.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        except TypeError:
            return self.tokenizer.decode(ids)

    def chunk_documents(self,documents):
        grouped=defaultdict(list)
        for document in documents:
            source=document["metadata"]["source"]
            grouped[source].append(document)#这里已经依据名称分开
        chunks=[]
        for source,pages in grouped.items():
            pages.sort( key=lambda x: x["metadata"]["page"])

            all_tokens=[]
            token_pages=[]

            for page in pages:

                text=page["text"]
                page_number=page["metadata"]["page"]
                if all_tokens:
                    sep_ids = self.encoder("\n\n")
                    all_tokens.extend(sep_ids)
                    token_pages.extend([page_number]*len(sep_ids))
    
                ids=self.encoder(text)
                all_tokens.extend(ids)
                token_pages.extend([page_number]*len(ids))
            
            if not all_tokens:
                continue

            start=0
            chunk_index=0

            stride=self.chunk_size - self.chunk_overlap

            while start<len(all_tokens):
                end=min(start+self.chunk_size-1,len(all_tokens))
                chunk_ids=all_tokens[start:end]
                chunk_text=self.decoder(chunk_ids).strip()
                page_numbers=sorted(
                    set(token_pages[start:end])
                )

                if chunk_text:
                    chunk={
                        "text":chunk_text,
                        "metadata": {
                            "source": source,

                            "chunk_id": (
                                f"{source}::chunk_{chunk_index}"
                            ),

                            "chunk_index": chunk_index,

                            "page_start": page_numbers[0],
                            "page_end": page_numbers[-1],
                            "pages": page_numbers,

                            "token_count": len(chunk_ids),

                            "token_start": start,
                            "token_end": end
                        }
                    }
                    chunks.append(chunk)
                  # 最后一个窗口已经覆盖全部文本
                if end == len(all_tokens):
                    break

                # 移动窗口
                start += stride
                chunk_index += 1
        return chunks
                 

if __name__ == "__main__":
    # 检查每个 Chunk 的 Token 数量
    from text_loader.pdf_loader import load_pdf
    docs = load_pdf(r"C:\Users\86182\Desktop\pytorch_learning\基于混合检索与重排序的本地知识库问答系统\data\raw\GPT-1.pdf")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    chunker=Chunker(tokenizer)
    chunks=chunker.chunk_documents(docs)
    print("总 Chunk 数量：", len(chunks))

    for chunk in chunks[:3]:

        print("\n========== CHUNK ==========")

        print(chunk["metadata"])

        print(chunk["text"][:500])



