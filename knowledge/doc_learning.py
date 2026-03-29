import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs", ".html", ".htm", ".css",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv",
    ".sql", ".sh", ".bat", ".xml", ".docx",
}

CN_STOPWORDS = {
    "我们", "你们", "他们", "这个", "那个", "以及", "一个", "一种", "已经", "进行", "可以", "如果",
    "因为", "所以", "没有", "自己", "通过", "需要", "其中", "然后", "还有", "这些", "那些", "一个",
}
EN_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "into", "your", "have", "will",
    "about", "there", "their", "which", "would", "could", "should", "been", "being",
    "than", "then", "also", "only", "such", "more", "most", "when", "what", "where",
}


def normalize_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def safe_relpath(path, root):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return os.path.basename(path)


def read_docx(path):
    with zipfile.ZipFile(path) as archive:
        xml_data = archive.read("word/document.xml")
    root = ET.fromstring(xml_data)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def read_document(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".docx":
        return read_docx(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def discover_files(input_path, recursive=True, extensions=None):
    input_path = Path(input_path)
    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in (extensions or DEFAULT_EXTENSIONS)}
    if input_path.is_file():
        return [str(input_path)] if input_path.suffix.lower() in extensions else []
    pattern = "**/*" if recursive else "*"
    files = []
    for path in sorted(input_path.glob(pattern)):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(str(path))
    return files


def split_sentences(text):
    chunks = re.split(r"(?<=[。！？!?；;.\n])", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def chunk_text(text, max_chars=1200, overlap=160):
    text = normalize_text(text)
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        paragraphs = split_sentences(text)
    chunks = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        sentences = split_sentences(paragraph)
        current = ""
        for sentence in sentences:
            candidate = f"{current}{sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence[-max_chars:]
        if current:
            current = current[-max_chars:]
    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    merged = []
    prev_tail = ""
    for chunk in chunks:
        if prev_tail:
            merged.append((prev_tail + "\n" + chunk).strip())
        else:
            merged.append(chunk)
        prev_tail = chunk[-overlap:]
    return merged


def pick_title(text, fallback):
    for line in normalize_text(text).splitlines():
        line = line.strip().lstrip("#").strip()
        if line.startswith("<") and line.endswith(">"):
            continue
        if line.startswith("![") or line.startswith("[!"):
            continue
        if len(line) >= 4:
            return line[:80]
    return fallback


def tokenize_keywords(text):
    text = text.lower()
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z][a-z0-9_\-]{2,}", text)
    return [tok for tok in tokens if tok not in CN_STOPWORDS and tok not in EN_STOPWORDS]


def extract_keywords(text, top_k=8):
    counter = Counter(tokenize_keywords(text))
    return [token for token, _ in counter.most_common(top_k)]


def extract_key_sentences(text, max_sentences=4):
    sentences = split_sentences(normalize_text(text))
    if len(sentences) <= max_sentences:
        return sentences
    keyword_weights = Counter(tokenize_keywords(text))
    scored = []
    for sentence in sentences:
        score = sum(keyword_weights.get(token, 0) for token in tokenize_keywords(sentence))
        score += min(len(sentence), 120) / 120
        scored.append((score, sentence))
    selected = [sentence for _, sentence in sorted(scored, key=lambda item: item[0], reverse=True)[:max_sentences]]
    ordered = [sentence for sentence in sentences if sentence in selected]
    return ordered[:max_sentences]


def build_knowledge_card(chunk, max_facts=4):
    facts = extract_key_sentences(chunk["text"], max_sentences=max_facts)
    keywords = extract_keywords(chunk["text"], top_k=8)
    card = {
        "source": chunk["source"],
        "title": chunk["title"],
        "chunk_id": chunk["chunk_id"],
        "keywords": keywords,
        "key_facts": facts,
    }
    return json.dumps(card, ensure_ascii=False, indent=2)


def build_grounded_answer(chunk, max_facts=4):
    facts = extract_key_sentences(chunk["text"], max_sentences=max_facts)
    lines = [
        f"来源：{chunk['source']}",
        f"标题：{chunk['title']}",
        "关键事实：",
    ]
    for fact in facts:
        lines.append(f"- {fact}")
    return "\n".join(lines)


def make_message(role, content):
    return {
        "role": role,
        "content": content,
        "reasoning_content": "",
        "tools": "",
        "tool_calls": "",
    }


def build_doc_sft_records(chunk, max_facts=4):
    knowledge_card_prompt = (
        "请阅读下面的文档片段，并构建结构化知识卡片。"
        "要求：只保留文档中明确出现的信息，不要补充常识，不要幻想。\n\n"
        f"文档来源：{chunk['source']}\n"
        f"标题：{chunk['title']}\n"
        f"片段内容：\n{chunk['text']}"
    )
    grounded_summary_prompt = (
        f"请基于文档《{chunk['title']}》当前片段，给出精准摘要并标注来源。"
        "如果片段无法支持某个结论，就不要补充。\n\n"
        f"文档来源：{chunk['source']}\n"
        f"片段内容：\n{chunk['text']}"
    )
    return [
        {"conversations": [make_message("user", knowledge_card_prompt), make_message("assistant", build_knowledge_card(chunk, max_facts=max_facts))]},
        {"conversations": [make_message("user", grounded_summary_prompt), make_message("assistant", build_grounded_answer(chunk, max_facts=max_facts))]},
    ]


def build_pretrain_record(chunk):
    return {
        "text": (
            f"文档来源：{chunk['source']}\n"
            f"标题：{chunk['title']}\n"
            f"片段编号：{chunk['chunk_id']}\n"
            f"内容：\n{chunk['text']}"
        )
    }


class LocalDocKnowledgeBase:
    def __init__(self, chunks, vectorizer, matrix):
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix

    @classmethod
    def from_path(cls, input_path, recursive=True, extensions=None, chunk_size=1200, chunk_overlap=160, min_chars=80):
        files = discover_files(input_path, recursive=recursive, extensions=extensions)
        chunks = []
        for file_path in files:
            try:
                text = normalize_text(read_document(file_path))
            except Exception:
                continue
            if len(text) < min_chars:
                continue
            title = pick_title(text, Path(file_path).stem)
            source = safe_relpath(file_path, input_path if Path(input_path).is_dir() else Path(file_path).parent)
            for chunk_id, chunk_text_item in enumerate(chunk_text(text, max_chars=chunk_size, overlap=chunk_overlap), start=1):
                if len(chunk_text_item.strip()) < min_chars:
                    continue
                chunks.append({
                    "source": source,
                    "title": title,
                    "path": str(file_path),
                    "chunk_id": chunk_id,
                    "text": chunk_text_item.strip(),
                })

        if not chunks:
            return cls([], None, None)

        corpus = [f"{chunk['title']}\n{chunk['source']}\n{chunk['text']}" for chunk in chunks]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50000)
        matrix = vectorizer.fit_transform(corpus)
        return cls(chunks, vectorizer, matrix)

    def is_ready(self):
        return bool(self.chunks) and self.vectorizer is not None and self.matrix is not None

    def stats(self):
        return {"chunks": len(self.chunks)}

    def search(self, query, top_k=4, min_score=0.05):
        if not self.is_ready() or not query.strip():
            return []
        query_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vec.T).toarray().ravel()
        ranked_ids = scores.argsort()[::-1]
        results = []
        for index in ranked_ids[: max(top_k * 3, top_k)]:
            score = float(scores[index])
            if score < min_score:
                continue
            chunk = dict(self.chunks[index])
            chunk["score"] = score
            results.append(chunk)
            if len(results) >= top_k:
                break
        return results

    def format_hits(self, query, top_k=4, min_score=0.05, max_chars=2200):
        hits = self.search(query, top_k=top_k, min_score=min_score)
        if not hits:
            return "", []
        blocks = []
        total_chars = 0
        kept = []
        for rank, hit in enumerate(hits, start=1):
            block = (
                f"[证据 {rank}] source={hit['source']} | title={hit['title']} | chunk={hit['chunk_id']} | score={hit['score']:.3f}\n"
                f"{hit['text']}"
            )
            if total_chars + len(block) > max_chars and blocks:
                break
            blocks.append(block)
            kept.append(hit)
            total_chars += len(block)
        return "\n\n".join(blocks), kept

    def build_system_prompt(self, query, top_k=4, min_score=0.05, max_chars=2200):
        context, hits = self.format_hits(query, top_k=top_k, min_score=min_score, max_chars=max_chars)
        if not context:
            return "", []
        prompt = (
            "以下是你必须优先参考的本地文档证据。"
            "回答时优先依据这些证据；如果证据不足，明确说“文档中没有足够依据”，不要编造。\n\n"
            f"{context}"
        )
        return prompt, hits
