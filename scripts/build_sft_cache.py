import argparse
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer

from dataset.lm_dataset import build_sft_cache, default_sft_cache_path


def main():
    parser = argparse.ArgumentParser(description="Build offline tensor cache for SFT training")
    parser.add_argument("--data_path", type=str, required=True, help="原始 SFT jsonl 数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="tokenizer 路径")
    parser.add_argument("--max_seq_len", type=int, default=768, help="最大序列长度")
    parser.add_argument("--cache_path", type=str, default="", help="输出缓存路径；为空则自动生成")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已有缓存")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    cache_path = args.cache_path or default_sft_cache_path(args.data_path, args.max_seq_len)
    result = build_sft_cache(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        cache_path=cache_path,
        overwrite=args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
