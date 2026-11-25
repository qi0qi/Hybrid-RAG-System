from pathlib import Path
import pymupdf

try:
    from .text_cleaner import clean_pdf_text
except ImportError:
    # 同时兼容 `python ingestion/pdf_loader.py` 直接运行。
    from text_cleaner import clean_pdf_text

def load_pdf(pdf_path):
    pdf_path = Path(pdf_path)

    documents = []

    with pymupdf.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):

            raw_text = page.get_text("text", sort=True)
            text=clean_pdf_text(raw_text)
            if not text.strip():
                continue

            document = {
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_index + 1
                }
            }

            documents.append(document)

    return documents


if __name__ == "__main__":
    docs = load_pdf(r"C:\Users\86182\Desktop\pytorch_learning\基于混合检索与重排序的本地知识库问答系统\data\raw\GPT-1.pdf")

    print("读取到的页面数量：", len(docs))
    print(docs[0])
