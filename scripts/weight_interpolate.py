"""
Offline weight interpolation between two or three HuggingFace models.

Usage:
    python scripts/weight_interpolate.py \
        --model_a /path/to/model_a \
        --model_b /path/to/model_b \
        --alpha 0.5 \
        --output_dir /path/to/output

    python scripts/weight_interpolate.py \
        --model_a /path/to/model_a \
        --model_b /path/to/model_b \
        --model_c /path/to/model_c \
        --alpha 0.3 \
        --beta 0.2 \
        --output_dir /path/to/output

Formula:
    dual-merge:   merged = model_a * (1 - alpha) + model_b * alpha
    tri-merge:    merged = model_a * (1 - alpha - beta) + model_b * alpha + model_c * beta
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


NON_WEIGHT_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
]


def load_sharded_safetensors(model_dir: str) -> dict[str, torch.Tensor]:
    """Load all safetensors shards from a HuggingFace model directory."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        # single file
        shard_files = ["model.safetensors"]

    state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(model_dir, shard_file)
        print(f"  Loading {shard_file}...")
        shard = load_file(shard_path)
        state_dict.update(shard)
    return state_dict


def _validate_state_dict_compatibility(
    reference_name: str,
    reference_state_dict: dict[str, torch.Tensor],
    other_name: str,
    other_state_dict: dict[str, torch.Tensor],
) -> None:
    """Ensure two state dicts have identical keys and shapes."""
    keys_ref = set(reference_state_dict.keys())
    keys_other = set(other_state_dict.keys())
    if keys_ref != keys_other:
        only_ref = keys_ref - keys_other
        only_other = keys_other - keys_ref
        raise ValueError(
            f"State dict keys mismatch.\n"
            f"  Only in {reference_name} ({len(only_ref)}): {sorted(only_ref)[:5]}...\n"
            f"  Only in {other_name} ({len(only_other)}): {sorted(only_other)[:5]}..."
        )

    for key in sorted(keys_ref):
        if reference_state_dict[key].shape != other_state_dict[key].shape:
            raise ValueError(
                f"Shape mismatch for key '{key}': "
                f"{reference_name}={reference_state_dict[key].shape} vs "
                f"{other_name}={other_state_dict[key].shape}"
            )


def interpolate_state_dicts(
    sd_a: dict[str, torch.Tensor],
    sd_b: dict[str, torch.Tensor],
    alpha: float,
    sd_c: dict[str, torch.Tensor] | None = None,
    beta: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Linear interpolation across two or three state dicts."""
    _validate_state_dict_compatibility("model_a", sd_a, "model_b", sd_b)
    if sd_c is not None:
        _validate_state_dict_compatibility("model_a", sd_a, "model_c", sd_c)

    keys_a = set(sd_a.keys())
    merged = {}
    weight_a = 1.0 - alpha - beta if sd_c is not None else 1.0 - alpha
    for key in sorted(keys_a):
        t_a = sd_a[key]
        t_b = sd_b[key]
        merged_tensor = t_a.float() * weight_a + t_b.float() * alpha
        if sd_c is not None:
            merged_tensor = merged_tensor + sd_c[key].float() * beta
        merged[key] = merged_tensor.to(t_a.dtype)
    return merged


def save_sharded_safetensors(
    state_dict: dict[str, torch.Tensor],
    output_dir: str,
    reference_model_dir: str,
):
    """Save state_dict using the same sharding layout as the reference model."""
    index_path = os.path.join(reference_model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        # Group keys by shard file
        shard_to_keys: dict[str, list[str]] = {}
        for key, shard_file in weight_map.items():
            shard_to_keys.setdefault(shard_file, []).append(key)

        # Save each shard
        for shard_file, keys in sorted(shard_to_keys.items()):
            shard_data = {k: state_dict[k] for k in keys}
            out_path = os.path.join(output_dir, shard_file)
            print(f"  Saving {shard_file} ({len(keys)} tensors)...")
            save_file(shard_data, out_path)

        # Copy the index file
        out_index = os.path.join(output_dir, "model.safetensors.index.json")
        shutil.copy2(index_path, out_index)
    else:
        # Single file
        out_path = os.path.join(output_dir, "model.safetensors")
        print(f"  Saving model.safetensors ({len(state_dict)} tensors)...")
        save_file(state_dict, out_path)


def main():
    parser = argparse.ArgumentParser(description="Linear interpolation between two or three HuggingFace models")
    parser.add_argument("--model_a", required=True, type=str, help="Path to HuggingFace model A")
    parser.add_argument("--model_b", required=True, type=str, help="Path to HuggingFace model B")
    parser.add_argument("--model_c", type=str, default=None, help="Optional path to HuggingFace model C")
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight for model B. Dual: merged = a*(1-alpha) + b*alpha"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Optional weight for model C. Tri: merged = a*(1-alpha-beta) + b*alpha + c*beta",
    )
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for merged model")
    args = parser.parse_args()

    assert 0.0 <= args.alpha <= 1.0, f"alpha must be in [0, 1], got {args.alpha}"
    assert 0.0 <= args.beta <= 1.0, f"beta must be in [0, 1], got {args.beta}"
    if args.model_c is None:
        assert args.beta == 0.0, f"beta must be 0 when model_c is not provided, got {args.beta}"
    else:
        assert args.alpha + args.beta <= 1.0, f"alpha + beta must be <= 1 for tri-merge, got {args.alpha + args.beta}"

    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")
    if args.model_c is not None:
        print(f"Model C: {args.model_c}")
    print(f"Alpha: {args.alpha}")
    if args.model_c is not None:
        weight_a = 1 - args.alpha - args.beta
        print(f"Beta: {args.beta}")
        print(f"Formula: merged = A * {weight_a:.2f} + B * {args.alpha:.2f} + C * {args.beta:.2f}")
    else:
        print(f"Formula: merged = A * {1 - args.alpha:.2f} + B * {args.alpha:.2f}")
    print(f"Output: {args.output_dir}")
    print()

    # Load
    print("Loading model A...")
    sd_a = load_sharded_safetensors(args.model_a)
    print(f"  Loaded {len(sd_a)} tensors")

    print("Loading model B...")
    sd_b = load_sharded_safetensors(args.model_b)
    print(f"  Loaded {len(sd_b)} tensors")

    sd_c = None
    if args.model_c is not None:
        print("Loading model C...")
        sd_c = load_sharded_safetensors(args.model_c)
        print(f"  Loaded {len(sd_c)} tensors")

    # Interpolate
    if args.model_c is not None:
        print(f"\nInterpolating with alpha={args.alpha}, beta={args.beta}...")
    else:
        print(f"\nInterpolating with alpha={args.alpha}...")
    merged = interpolate_state_dicts(sd_a, sd_b, args.alpha, sd_c=sd_c, beta=args.beta)
    del sd_a, sd_b, sd_c
    print(f"  Merged {len(merged)} tensors")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy non-weight files from model_a
    print("\nCopying config/tokenizer files from model A...")
    for filename in NON_WEIGHT_FILES:
        src = os.path.join(args.model_a, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, filename))
            print(f"  Copied {filename}")

    print("\nSaving merged weights...")
    save_sharded_safetensors(merged, args.output_dir, args.model_a)

    print(f"\nDone! Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
