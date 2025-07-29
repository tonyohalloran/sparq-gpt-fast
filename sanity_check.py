#!/usr/bin/env python3
"""
Sanity check script to verify SparQ activation on/off.
Loads model and runs a simple generation to verify SparQ status.
"""

import argparse
import sys
from pathlib import Path

import torch
from benchmark import load_model, forward_for_generate, prefill, sample
from model import Transformer
from sparq import SparQArgs, RKForCompressionRatio, SparQAttention
from tokenizer import get_tokenizer


def check_sparq_active(model: Transformer) -> tuple[bool, int]:
    """Check if SparQ is active and return compression ratio."""
    attention_func = model.layers[0].attention.attention_function
    if isinstance(attention_func, SparQAttention):
        return True, attention_func.config.rk.ratio
    return False, 0


def main():
    parser = argparse.ArgumentParser(description="SparQ sanity check")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--use-sparq", action="store_true", help="Enable SparQ attention")
    parser.add_argument("--no-sparq", action="store_true", help="Disable SparQ attention")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()

    if args.use_sparq and args.no_sparq:
        print("Error: Cannot specify both --use-sparq and --no-sparq")
        sys.exit(1)
    
    if not args.use_sparq and not args.no_sparq:
        print("Error: Must specify either --use-sparq or --no-sparq")
        sys.exit(1)

    # Check paths
    if not args.model.is_file():
        print(f"Error: Model checkpoint not found: {args.model}")
        sys.exit(1)
    
    tokenizer_path = args.model.parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        print(f"Error: Tokenizer not found: {tokenizer_path}")
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    attention_mode = "sparq" if args.use_sparq else "dense"
    sparq_config = SparQArgs(rk=RKForCompressionRatio(8)) if args.use_sparq else SparQArgs()
    
    model = load_model(
        args.model,
        device,
        precision=torch.bfloat16,
        attention=attention_mode,
        sparq=sparq_config,
    )
    
    # Check SparQ status
    sparq_active, compression = check_sparq_active(model)
    print(f"Device: {device}; SparQ active: {sparq_active}; compression={compression}")
    
    # Verify expected state
    if args.use_sparq and not sparq_active:
        print("ERROR: Expected SparQ to be active but it's not!")
        sys.exit(1)
    elif args.no_sparq and sparq_active:
        print("ERROR: Expected SparQ to be inactive but it's active!")
        sys.exit(1)

    # Load tokenizer and run a simple generation
    tokenizer = get_tokenizer(tokenizer_path, args.model)
    
    # Simple test prompt
    test_prompt = "The capital of France is"
    encoded_prompt = torch.tensor(tokenizer.encode(test_prompt), device=device, dtype=torch.int32)
    
    # Setup cache
    max_seq_len = len(encoded_prompt) + 10
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_len)
    
    # Generate a few tokens
    print(f"\nTest generation with prompt: '{test_prompt}'")
    input_pos = torch.arange(0, encoded_prompt.size(0), device=device)
    
    with torch.no_grad():
        # Prefill
        next_token = prefill(model, encoded_prompt.view(1, -1), input_pos, temperature=0)
        generated_tokens = [next_token.item()]
        
        # Generate a few more tokens
        for i in range(5):
            input_pos = torch.tensor([encoded_prompt.size(0) + i], device=device, dtype=torch.int)
            logits = forward_for_generate(model, next_token.view(1, -1), input_pos)
            next_token, _ = sample(logits, temperature=0, top_k=1)
            generated_tokens.append(next_token.item())
    
    # Decode and print result
    full_tokens = tokenizer.encode(test_prompt) + generated_tokens
    result = tokenizer.decode(full_tokens)
    print(f"Generated: {result}")
    print("\nSanity check completed successfully!")


if __name__ == "__main__":
    main() 