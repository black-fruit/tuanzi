import argparse
import json
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from knowledge.doc_learning import LocalDocKnowledgeBase, build_doc_sft_records, build_pretrain_record

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_PRETRAIN_OUT = os.path.join(REPO_ROOT, "dataset", "doc_pretrain.jsonl")
DEFAULT_SFT_OUT = os.path.join(REPO_ROOT, "dataset", "doc_sft.jsonl")


def write_jsonl(path, records):
    if not path:
        return 0
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Build document-grounded datasets for Tuanzi / MiniMind")
    parser.add_argument("--input_path", type=str, required=True, help="文档目录或单个文件路径")
    parser.add_argument("--recursive", type=int, default=1, help="是否递归扫描子目录（0/1）")
    parser.add_argument("--chunk_size", type=int, default=1200, help="每个文档块最大字符数")
    parser.add_argument("--chunk_overlap", type=int, default=160, help="相邻文档块的重叠字符数")
    parser.add_argument("--min_chars", type=int, default=80, help="最小有效文档块长度")
    parser.add_argument("--max_facts", type=int, default=4, help="每个知识卡片保留的关键事实数")
    parser.add_argument("--pretrain_out", type=str, default=DEFAULT_PRETRAIN_OUT, help="继续预训练语料输出路径")
    parser.add_argument("--sft_out", type=str, default=DEFAULT_SFT_OUT, help="文档监督学习数据输出路径")
    args = parser.parse_args()

    kb = LocalDocKnowledgeBase.from_path(
        args.input_path,
        recursive=bool(args.recursive),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chars=args.min_chars,
    )
    if not kb.chunks:
        raise SystemExit("没有找到可用文档块，请检查输入路径或文件类型")

    pretrain_records = (build_pretrain_record(chunk) for chunk in kb.chunks)
    sft_records = (
        record
        for chunk in kb.chunks
        for record in build_doc_sft_records(chunk, max_facts=args.max_facts)
    )

    os.makedirs(os.path.dirname(args.pretrain_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.sft_out), exist_ok=True)
    pretrain_count = write_jsonl(args.pretrain_out, pretrain_records)
    sft_count = write_jsonl(args.sft_out, sft_records)
    print(
        f"文档块: {len(kb.chunks)} | 预训练样本: {pretrain_count} -> {args.pretrain_out} | "
        f"SFT样本: {sft_count} -> {args.sft_out}"
    )


if __name__ == "__main__":
    main()
