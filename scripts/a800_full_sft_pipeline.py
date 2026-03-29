import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd, cwd=REPO_ROOT, env=None, check=True):
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), env=env, text=True)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def detect_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=True,
        )
        line = result.stdout.strip().splitlines()[0]
        name, memory = [part.strip() for part in line.split(",")]
        return name, memory
    except Exception:
        return "unknown", "unknown"


def find_first(root, filename):
    root = Path(root)
    matches = list(root.rglob(filename))
    return str(matches[0]) if matches else ""


def ensure_modelscope_download(kind, repo_id, include_file, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    target = find_first(local_dir, include_file)
    if target:
        return target
    cmd = ["modelscope", "download", f"--{kind}", repo_id, "--include", include_file, "--local_dir", local_dir]
    run(cmd)
    target = find_first(local_dir, include_file)
    if not target:
        raise FileNotFoundError(f"Failed to download {include_file} from {repo_id}")
    return target


def compute_num_workers():
    cpu_count = os.cpu_count() or 8
    return max(4, min(16, cpu_count // 2))


def merge_doc_dataset(base_sft_path, doc_sft_path, output_path):
    merge_jsonl_files([base_sft_path, doc_sft_path], output_path)


def merge_jsonl_files(inputs, output_path):
    cmd = [sys.executable, "scripts/merge_jsonl.py", "--inputs"]
    cmd.extend([str(path) for path in inputs])
    cmd.extend(["--output", str(output_path)])
    run(cmd)


def run_full_sft(
    data_path,
    save_weight,
    from_weight,
    epochs,
    learning_rate,
    batch_size,
    accumulation_steps,
    dtype,
    num_workers,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    num_key_value_heads,
    max_seq_len,
    max_position_embeddings,
    cache_data,
    cache_path,
    use_compile,
    log_interval=50,
    save_interval=10**9,
    intermediate_size=None,
):
    train_cmd = [
        sys.executable,
        "trainer/train_full_sft.py",
        "--data_path",
        str(data_path),
        "--save_weight",
        save_weight,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(learning_rate),
        "--dtype",
        dtype,
        "--num_workers",
        str(num_workers),
        "--pin_memory",
        "1",
        "--persistent_workers",
        "1",
        "--prefetch_factor",
        "4",
        "--accumulation_steps",
        str(accumulation_steps),
        "--hidden_size",
        str(hidden_size),
        "--num_hidden_layers",
        str(num_hidden_layers),
        "--num_attention_heads",
        str(num_attention_heads),
        "--num_key_value_heads",
        str(num_key_value_heads),
        "--max_seq_len",
        str(max_seq_len),
        "--max_position_embeddings",
        str(max_position_embeddings),
        "--from_weight",
        from_weight,
        "--cache_data",
        str(cache_data),
        "--cache_path",
        str(cache_path),
        "--use_compile",
        str(use_compile),
        "--log_interval",
        str(log_interval),
        "--save_interval",
        str(save_interval),
    ]
    if intermediate_size:
        train_cmd.extend(["--intermediate_size", str(intermediate_size)])
    run(train_cmd)


def main():
    parser = argparse.ArgumentParser(description="A800-40G optimized full SFT automation pipeline for Tuanzi")
    parser.add_argument("--base_sft_file", default="sft_t2t_mini.jsonl", type=str, help="基础 SFT 数据文件名")
    parser.add_argument("--doc_path", default="docs", type=str, help="本地文档目录")
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--num_key_value_heads", default=4, type=int)
    parser.add_argument("--intermediate_size", default=None, type=int)
    parser.add_argument("--max_position_embeddings", default=32768, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--batch_size", default=160, type=int, help="A800-40G 1小时模式默认 batch size")
    parser.add_argument("--accumulation_steps", default=1, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--learning_rate", default=1.5e-5, type=float)
    parser.add_argument("--save_weight", default="full_sft_a800_1h", type=str)
    parser.add_argument("--bio_mode", default=1, choices=[0, 1], type=int, help="是否启用仿生双阶段训练")
    parser.add_argument("--doc_focus_epochs", default=1, type=int, help="文档快记忆阶段轮数")
    parser.add_argument("--doc_focus_learning_rate", default=3e-5, type=float, help="文档快记忆阶段学习率")
    parser.add_argument("--doc_replay_factor", default=2, type=int, help="慢整合阶段文档回放倍数")
    parser.add_argument("--cache_data", default=1, choices=[0, 1], type=int)
    parser.add_argument("--use_compile", default=1, choices=[0, 1], type=int)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"], type=str)
    parser.add_argument("--time_profile", default="1h", choices=["1h", "balanced"], type=str, help="训练时长目标配置")
    parser.add_argument("--upload_kaggle", default=1, choices=[0, 1], type=int)
    parser.add_argument("--kaggle_owner", default="black-fruit", type=str, help="Kaggle owner slug，可覆盖")
    parser.add_argument("--kaggle_model_slug", default="tuanzi-a800-full-sft", type=str)
    parser.add_argument("--kaggle_instance_slug", default="transformers-base", type=str)
    args = parser.parse_args()

    gpu_name, gpu_memory = detect_gpu()
    print(f"[GPU] {gpu_name} | {gpu_memory}")
    if "A800" not in gpu_name.upper():
        print("[WARN] Current GPU is not A800-40G; pipeline will still run with the same defaults.")
    if args.time_profile == "1h":
        print("[PROFILE] 1h profile enabled: short sequence, single main epoch, larger batch, reduced replay.")

    dataset_dir = REPO_ROOT / "dataset"
    out_dir = REPO_ROOT / "out"
    cache_dir = REPO_ROOT / ".cache" / "modelscope"
    export_dir = REPO_ROOT / "exports" / args.save_weight
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    base_sft = ensure_modelscope_download("dataset", "gongjy/minimind_dataset", args.base_sft_file, str(dataset_dir))
    pretrain_weight = ensure_modelscope_download("model", "gongjy/minimind-3-pytorch", f"pretrain_{args.hidden_size}.pth", str(cache_dir / "minimind-3-pytorch"))
    shutil.copy2(pretrain_weight, out_dir / f"pretrain_{args.hidden_size}.pth")

    doc_path = REPO_ROOT / args.doc_path
    run(
        [
            sys.executable,
            "scripts/build_doc_dataset.py",
            "--input_path",
            str(doc_path),
            "--pretrain_out",
            str(dataset_dir / "doc_pretrain.jsonl"),
            "--sft_out",
            str(dataset_dir / "doc_sft.jsonl"),
        ]
    )

    merged_sft = dataset_dir / f"{Path(args.base_sft_file).stem}.with_docs.jsonl"
    num_workers = compute_num_workers()
    doc_sft_path = dataset_dir / "doc_sft.jsonl"
    merge_inputs = [base_sft]
    merge_inputs.extend([str(doc_sft_path)] * max(1, args.doc_replay_factor))
    merge_jsonl_files(merge_inputs, merged_sft)

    base_weight = "pretrain"
    if args.bio_mode == 1:
        doc_focus_weight = f"{args.save_weight}_doc_focus"
        print(
            "[BIO] complementary-learning mode enabled: "
            "fast doc memory stage (hippocampus-like) -> replay consolidation stage (cortex-like)"
        )
        run_full_sft(
            data_path=doc_sft_path,
            save_weight=doc_focus_weight,
            from_weight=base_weight,
            epochs=args.doc_focus_epochs,
            learning_rate=args.doc_focus_learning_rate,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            dtype=args.dtype,
            num_workers=num_workers,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            max_seq_len=args.max_seq_len,
            max_position_embeddings=args.max_position_embeddings,
            cache_data=args.cache_data,
            cache_path=dataset_dir / f"{doc_sft_path.stem}.bio.cache.pt",
            use_compile=args.use_compile,
            log_interval=20,
            intermediate_size=args.intermediate_size,
        )
        base_weight = doc_focus_weight

    run_full_sft(
        data_path=merged_sft,
        save_weight=args.save_weight,
        from_weight=base_weight,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        dtype=args.dtype,
        num_workers=num_workers,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_seq_len=args.max_seq_len,
        max_position_embeddings=args.max_position_embeddings,
        cache_data=args.cache_data,
        cache_path=dataset_dir / f"{merged_sft.stem}.cache.pt",
        use_compile=args.use_compile,
        log_interval=50,
        intermediate_size=args.intermediate_size,
    )

    checkpoint_path = out_dir / f"{args.save_weight}_{args.hidden_size}.pth"
    export_cmd = [
        sys.executable,
        "scripts/export_transformers_model.py",
        "--torch_path",
        str(checkpoint_path),
        "--output_dir",
        str(export_dir),
        "--tokenizer_path",
        str(REPO_ROOT / "model"),
        "--dtype",
        args.dtype,
        "--hidden_size",
        str(args.hidden_size),
        "--num_hidden_layers",
        str(args.num_hidden_layers),
        "--num_attention_heads",
        str(args.num_attention_heads),
        "--num_key_value_heads",
        str(args.num_key_value_heads),
        "--max_position_embeddings",
        str(args.max_position_embeddings),
    ]
    if args.intermediate_size:
        export_cmd.extend(["--intermediate_size", str(args.intermediate_size)])
    run(export_cmd)

    if args.upload_kaggle == 1:
        run(
            [
                sys.executable,
                "scripts/upload_kaggle_model.py",
                "--export_dir",
                str(export_dir),
                "--owner_slug",
                args.kaggle_owner,
                "--model_slug",
                args.kaggle_model_slug,
                "--model_title",
                "Tuanzi A800 Full SFT",
                "--instance_slug",
                args.kaggle_instance_slug,
                "--version_notes",
                "A800-40G optimized full SFT with local doc learning",
                "--training_data",
                "gongjy/minimind_dataset",
                "docs/basic_math_science.md",
            ]
        )

    print(f"[DONE] checkpoint={checkpoint_path}")
    print(f"[DONE] export_dir={export_dir}")


if __name__ == "__main__":
    main()
