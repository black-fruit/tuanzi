import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, env=None, check=True):
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def ensure_kaggle_env(args):
    env = os.environ.copy()
    if args.owner_slug and "KAGGLE_USERNAME" not in env:
        env["KAGGLE_USERNAME"] = args.owner_slug
    if "KAGGLE_KEY" not in env and env.get("KAGGLE_API_TOKEN"):
        env["KAGGLE_KEY"] = env["KAGGLE_API_TOKEN"]
    if "KAGGLE_USERNAME" not in env:
        raise RuntimeError("Kaggle owner slug is required. Set --owner_slug or export KAGGLE_USERNAME.")
    if "KAGGLE_KEY" not in env:
        raise RuntimeError("Kaggle auth key is required. Export KAGGLE_KEY, or provide KAGGLE_API_TOKEN if your environment maps it.")
    return env


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def prepare_metadata(args, export_dir, owner_slug):
    model_metadata = {
        "ownerSlug": owner_slug,
        "title": args.model_title,
        "slug": args.model_slug,
        "subtitle": args.model_subtitle,
        "isPrivate": bool(args.is_private),
        "description": args.model_description,
        "publishTime": "",
        "provenanceSources": args.provenance_sources,
    }
    instance_metadata = {
        "ownerSlug": owner_slug,
        "modelSlug": args.model_slug,
        "instanceSlug": args.instance_slug,
        "framework": args.framework,
        "overview": args.instance_overview,
        "usage": args.instance_usage,
        "licenseName": args.license_name,
        "fineTunable": True,
        "trainingData": args.training_data,
        "modelInstanceType": "Unspecified",
        "baseModelInstance": "",
        "externalBaseModelUrl": "",
    }
    write_json(os.path.join(export_dir, "model-metadata.json"), model_metadata)
    write_json(os.path.join(export_dir, "model-instance-metadata.json"), instance_metadata)


def ensure_model_and_instance(args, export_dir, env):
    owner_slug = env["KAGGLE_USERNAME"]
    model_ref = f"{owner_slug}/{args.model_slug}"
    instance_ref = f"{owner_slug}/{args.model_slug}/{args.framework}/{args.instance_slug}"
    prepare_metadata(args, export_dir, owner_slug)

    model_get = run(["kaggle", "models", "get", model_ref], env=env, check=False)
    if model_get.returncode != 0:
        run(["kaggle", "models", "create", "-p", export_dir], env=env)
    else:
        run(["kaggle", "models", "update", "-p", export_dir], env=env)

    instance_get = run(["kaggle", "models", "instances", "get", instance_ref], env=env, check=False)
    if instance_get.returncode != 0:
        run(["kaggle", "models", "instances", "create", "-p", export_dir, "--dir-mode", "skip"], env=env)
    else:
        run(["kaggle", "models", "instances", "update", "-p", export_dir], env=env)
        run(
            [
                "kaggle",
                "models",
                "instances",
                "versions",
                "create",
                instance_ref,
                "-p",
                export_dir,
                "-n",
                args.version_notes,
                "--dir-mode",
                "skip",
            ],
            env=env,
        )
    print(instance_ref)


def main():
    parser = argparse.ArgumentParser(description="Upload a Transformers model directory to Kaggle Models")
    parser.add_argument("--export_dir", required=True, type=str, help="Transformers 模型目录")
    parser.add_argument("--owner_slug", default="", type=str, help="Kaggle 用户名；为空则读取 KAGGLE_USERNAME")
    parser.add_argument("--model_slug", default="tuanzi-a800-full-sft", type=str)
    parser.add_argument("--model_title", default="Tuanzi A800 Full SFT", type=str)
    parser.add_argument("--model_subtitle", default="A800-40G optimized full SFT export with local doc learning", type=str)
    parser.add_argument("--model_description", default="# Model Summary\n\nA Tuanzi / MiniMind export trained through the A800 full SFT automation pipeline.\n", type=str)
    parser.add_argument("--instance_slug", default="transformers-base", type=str)
    parser.add_argument("--framework", default="transformers", type=str)
    parser.add_argument("--instance_overview", default="Transformers export of the A800 optimized full SFT checkpoint.", type=str)
    parser.add_argument("--instance_usage", default="# Usage\n\nLoad with `AutoTokenizer` and `AutoModelForCausalLM`.\n", type=str)
    parser.add_argument("--license_name", default="Apache 2.0", type=str)
    parser.add_argument("--training_data", nargs="*", default=["gongjy/minimind_dataset", "local-doc-learning"], help="训练数据来源描述")
    parser.add_argument("--provenance_sources", default="", type=str)
    parser.add_argument("--version_notes", default="A800 automation pipeline export", type=str)
    parser.add_argument("--is_private", default=1, choices=[0, 1], type=int)
    args = parser.parse_args()

    export_dir = os.path.abspath(args.export_dir)
    if not os.path.isdir(export_dir):
        raise SystemExit(f"export_dir not found: {export_dir}")
    env = ensure_kaggle_env(args)
    ensure_model_and_instance(args, export_dir, env)


if __name__ == "__main__":
    main()
