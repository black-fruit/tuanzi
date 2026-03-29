import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Merge multiple jsonl files into one")
    parser.add_argument("--inputs", nargs="+", required=True, help="输入 jsonl 文件列表")
    parser.add_argument("--output", required=True, help="输出 jsonl 文件")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    total = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for path in args.inputs:
            if not path or not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    fout.write(line + "\n")
                    total += 1
    print(f"{args.output}\t{total}")


if __name__ == "__main__":
    main()
