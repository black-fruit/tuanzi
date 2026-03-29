import argparse
import json
import os
import sys

import torch
import transformers
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainer.trainer_utils import build_lm_config_from_args


def export_checkpoint(args):
    lm_config = build_lm_config_from_args(args)
    state_dict = torch.load(args.torch_path, map_location="cpu")

    common_config = {
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "intermediate_size": lm_config.intermediate_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "num_key_value_heads": lm_config.num_key_value_heads,
        "head_dim": lm_config.hidden_size // lm_config.num_attention_heads,
        "max_position_embeddings": lm_config.max_position_embeddings,
        "rms_norm_eps": lm_config.rms_norm_eps,
        "rope_theta": lm_config.rope_theta,
        "tie_word_embeddings": True,
    }

    if not lm_config.use_moe:
        config = Qwen3Config(**common_config, use_sliding_window=False, sliding_window=None)
        model = Qwen3ForCausalLM(config)
    else:
        config = Qwen3MoeConfig(
            **common_config,
            num_experts=lm_config.num_experts,
            num_experts_per_tok=lm_config.num_experts_per_tok,
            moe_intermediate_size=lm_config.moe_intermediate_size,
            norm_topk_prob=lm_config.norm_topk_prob,
        )
        model = Qwen3MoeForCausalLM(config)
        if int(transformers.__version__.split(".")[0]) >= 5:
            new_sd = {k: v for k, v in state_dict.items() if "experts." not in k or "gate.weight" in k}
            for layer_id in range(lm_config.num_hidden_layers):
                prefix = f"model.layers.{layer_id}.mlp.experts"
                new_sd[f"{prefix}.gate_up_proj"] = torch.cat(
                    [
                        torch.stack([state_dict[f"{prefix}.{expert}.gate_proj.weight"] for expert in range(lm_config.num_experts)]),
                        torch.stack([state_dict[f"{prefix}.{expert}.up_proj.weight"] for expert in range(lm_config.num_experts)]),
                    ],
                    dim=1,
                )
                new_sd[f"{prefix}.down_proj"] = torch.stack(
                    [state_dict[f"{prefix}.{expert}.down_proj.weight"] for expert in range(lm_config.num_experts)]
                )
            state_dict = new_sd

    model.load_state_dict(state_dict, strict=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = model.to(dtype)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.save_pretrained(args.output_dir)

    if int(transformers.__version__.split(".")[0]) >= 5:
        tokenizer_config_path = os.path.join(args.output_dir, "tokenizer_config.json")
        config_path = os.path.join(args.output_dir, "config.json")
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
        tokenizer_config["extra_special_tokens"] = {}
        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        model_config["rope_theta"] = lm_config.rope_theta
        model_config["rope_scaling"] = None
        if "rope_parameters" in model_config:
            del model_config["rope_parameters"]
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)

    print(args.output_dir)


def main():
    parser = argparse.ArgumentParser(description="Export a torch checkpoint to a Transformers-compatible directory")
    parser.add_argument("--torch_path", required=True, type=str, help="torch 权重路径，例如 out/full_sft_768.pth")
    parser.add_argument("--output_dir", required=True, type=str, help="Transformers 导出目录")
    parser.add_argument("--tokenizer_path", default="../model", type=str, help="tokenizer 目录")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"], help="导出权重精度")
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--num_key_value_heads", default=4, type=int)
    parser.add_argument("--intermediate_size", default=None, type=int)
    parser.add_argument("--max_position_embeddings", default=32768, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--use_moe", default=0, choices=[0, 1], type=int)
    args = parser.parse_args()
    export_checkpoint(args)


if __name__ == "__main__":
    main()
