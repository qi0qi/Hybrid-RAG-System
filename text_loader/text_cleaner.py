import re
import unicodedata


# PDF 中常见的 Unicode 连字
LIGATURES = str.maketrans({
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
})


def is_heading(line: str) -> bool:
    """判断是否为常见的论文章节标题。"""

    # 例如：1 Introduction、3.1 Unsupervised pre-training
    if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Za-z]", line):
        return True

    # 常见的无编号章节
    if line.lower() in {
        "abstract",
        "introduction",
        "conclusion",
        "conclusions",
        "references",
        "acknowledgements",
        "acknowledgments",
        "appendix",
    }:
        return True

    return False


def is_list_item(line: str) -> bool:
    """检测常见列表项。"""
    return bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", line))


def clean_pdf_text(text: str) -> str:
    """
    清洗 PDF 提取出的普通文本。

    主要功能：
    1. Unicode 规范化
    2. 删除不可见字符
    3. 合并多余空格
    4. 修复跨行单词
    5. 合并段落内部的换行
    6. 尽量保留标题、列表和段落结构
    """

    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")

    if not text.strip():
        return ""

    # ---------- 1. Unicode 规范化 ----------
    text = unicodedata.normalize("NFC", text)
    text = text.translate(LIGATURES)

    # ---------- 2. 清理特殊字符 ----------
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")
    text = text.replace("\t", " ")

    # 删除常见不可见格式字符
    text = re.sub(r"[\u200b\u2060\ufeff]", "", text)

    # ---------- 3. 规范每一行 ----------
    lines = []

    for line in text.split("\n"):
        line = re.sub(r" +", " ", line)
        lines.append(line.strip())

    # ---------- 4. 恢复段落结构 ----------
    paragraphs = []
    current = ""

    def flush():
        nonlocal current
        if current:
            paragraphs.append(current)
            current = ""

    for line in lines:

        # 原始空行：视作段落边界
        if not line:
            flush()
            continue

        # 章节标题单独保留
        if is_heading(line):
            flush()
            paragraphs.append(line)
            continue

        # 列表项单独开始
        if is_list_item(line):
            flush()
            current = line
            continue

        # 当前段落为空
        if not current:
            current = line
            continue

        # 上一行以连字符结尾，且下一行以字母开头
        if current.endswith("-") and re.match(r"^[A-Za-z]", line):
            current += line
        else:
            # 普通换行通常是 PDF 排版造成的
            current += " " + line

    flush()

    # ---------- 5. 规范段落间距 ----------
    result = "\n\n".join(paragraphs)

    return result.strip()
