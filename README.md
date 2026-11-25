# 数据结构

## `docs`

```python
docs = [
    {
        "text": "页面文本内容",
        "metadata": {
            "source": "GPT-1.pdf",
            "page": 1
        }
    },
    {
        "text": "页面文本内容",
        "metadata": {
            "source": "GPT-1.pdf",
            "page": 2
        }
    }
]
```

## `chunks`

```python
chunks = [
    {
        "text": "文本块内容",
        "metadata": {
            "source": "GPT-1.pdf",
            "chunk_id": "GPT-1.pdf::chunk_0",
            "chunk_index": 0,
            "page_start": 1,
            "page_end": 2,
            "pages": [1, 2],
            "token_count": 512,
            "token_start": 0,
            "token_end": 512
        }
    },
    {
        "text": "文本块内容",
        "metadata": {
            "source": "GPT-1.pdf",
            "chunk_id": "GPT-1.pdf::chunk_1",
            "chunk_index": 1,
            "page_start": 2,
            "page_end": 3,
            "pages": [2, 3],
            "token_count": 512,
            "token_start": 448,
            "token_end": 960
        }
    }
]
```
