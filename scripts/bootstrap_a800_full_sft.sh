#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/black-fruit/tuanzi"
WORKDIR="${HOME}/tuanzi-a800"
REPO_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
KAGGLE_OWNER="black-fruit"
KAGGLE_MODEL_SLUG="tuanzi-a800-full-sft"
UPLOAD_KAGGLE="1"
SKIP_TORCH_INSTALL="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --kaggle-owner)
      KAGGLE_OWNER="$2"
      shift 2
      ;;
    --kaggle-model-slug)
      KAGGLE_MODEL_SLUG="$2"
      shift 2
      ;;
    --upload-kaggle)
      UPLOAD_KAGGLE="$2"
      shift 2
      ;;
    --skip-torch-install)
      SKIP_TORCH_INSTALL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$WORKDIR"
if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="${WORKDIR}/repo"
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --ff-only
fi

cd "$REPO_DIR"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" - <<'PY'
import importlib
import os
import sys

def safe_version(name):
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return "not-installed"

py = sys.version_info
if py < (3, 10):
    raise SystemExit(f"Python >= 3.10 is required, got {sys.version.split()[0]}")

torch_version = safe_version("torch")
torchvision_version = safe_version("torchvision")
torchaudio_version = safe_version("torchaudio")
cuda_version = "unknown"
if torch_version != "not-installed":
    import torch
    cuda_version = getattr(torch.version, "cuda", None) or "cpu"

conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
conda_prefix = os.environ.get("CONDA_PREFIX", "")
prefix = f"{conda_env} ({conda_prefix})" if conda_env or conda_prefix else "none"

print(f"[ENV] python={sys.version.split()[0]} conda={prefix}")
print(f"[ENV] torch={torch_version} torchvision={torchvision_version} torchaudio={torchaudio_version} cuda={cuda_version}")
if torch_version != "not-installed" and not torch_version.startswith("2.5"):
    print(f"[WARN] Expected preinstalled torch 2.5.x, got {torch_version}")
if cuda_version not in {"unknown", "cpu"} and not str(cuda_version).startswith("12.4"):
    print(f"[WARN] Expected CUDA 12.4 runtime, got {cuda_version}")
PY

if [[ "$SKIP_TORCH_INSTALL" == "1" ]]; then
  if ! "$PYTHON_BIN" -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
    echo "torch is not available in the current environment. This bootstrap skips torch installation by default." >&2
    echo "Install torch in the server image first, or rerun with --skip-torch-install 0." >&2
    exit 1
  fi

  FILTERED_REQUIREMENTS="$(mktemp)"
  trap 'rm -f "$FILTERED_REQUIREMENTS"' EXIT
  "$PYTHON_BIN" - <<'PY' requirements.txt "$FILTERED_REQUIREMENTS"
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
pattern = re.compile(r"^\s*(torch|torchvision|torchaudio)(\b|[<>=!~])", re.IGNORECASE)
with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        if pattern.match(line):
            continue
        fout.write(line)
PY
  "$PYTHON_BIN" -m pip install -r "$FILTERED_REQUIREMENTS"
else
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

if [[ -n "${KAGGLE_API_TOKEN:-}" && -z "${KAGGLE_KEY:-}" ]]; then
  export KAGGLE_KEY="$KAGGLE_API_TOKEN"
fi
if [[ -n "$KAGGLE_OWNER" && -z "${KAGGLE_USERNAME:-}" ]]; then
  export KAGGLE_USERNAME="$KAGGLE_OWNER"
fi

"$PYTHON_BIN" scripts/a800_full_sft_pipeline.py \
  --time_profile 1h \
  --upload_kaggle "$UPLOAD_KAGGLE" \
  --kaggle_owner "$KAGGLE_OWNER" \
  --kaggle_model_slug "$KAGGLE_MODEL_SLUG"
