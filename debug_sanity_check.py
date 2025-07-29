#!/usr/bin/env python3
"""
Comprehensive sanity check script to debug LLM output corruption.
Follows the debugging plan to isolate tokenizer/model issues.
"""

import argparse
import json
import sys
import torch
import subprocess
from pathlib import Path

from tokenizer import get_tokenizer
from benchmark import load_model


def save_system_info():
    """Step 0: Record environment information"""
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'git_branch': subprocess.check_output(['git', 'branch', '--show-current']).decode().strip()
    }
    
    Path('outputs').mkdir(exist_ok=True)
    with open('outputs/system_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("=== System Info ===")
    print(json.dumps(info, indent=2))
    print()


def step1_tokenizer_sanity_check(model_path, device):
    """Step 1: Verify the prepared checkpoint + tokenizer outside our scripts"""
    print("=== Step 1: Tokenizer Sanity Check ===")
    
    # Load tokenizer
    tokenizer_path = model_path / "tokenizer.model"
    print(f"Tokenizer path: {tokenizer_path}")
    
    tokenizer = get_tokenizer(tokenizer_path, model_path)
    
    # Round-trip sanity check
    text = "Hello world! This is a short test."
    ids = tokenizer.encode(text)
    back = tokenizer.decode(ids)
    
    print(f"Original text: {text}")
    print(f"Encoded IDs: {ids[:10]}...")  # Show first 10 tokens
    print(f"Decoded back: {back}")
    print(f"Round-trip ok: {text[:10] in back}")
    print(f"eos_id: {tokenizer.eos_id()}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    
    # Check for special tokens
    if hasattr(tokenizer, 'special_tokens'):
        print(f"Special tokens: {list(tokenizer.special_tokens.keys())[:5]}...")
    
    print()


def step1_model_sanity_check(model_path, device):
    """Step 1: Verify model can generate normal text"""
    print("=== Step 1: Model Generation Sanity Check ===")
    
    # Load model
    print("Loading model...")
    model = load_model(Path(model_path), device, torch.bfloat16)
    
    # Load tokenizer
    tokenizer_path = model_path / "tokenizer.model"
    tokenizer = get_tokenizer(tokenizer_path, model_path)
    
    # Simple generation test
    prompt = "Write a haiku about the ocean."
    print(f"Prompt: {prompt}")
    
    # Encode prompt
    encoded = torch.tensor(tokenizer.encode(prompt), device=device, dtype=torch.int32)
    
    # Setup cache
    max_seq_len = len(encoded) + 64
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_len)
    
    # Generate
    input_pos = torch.arange(0, encoded.size(0), device=device)
    
    with torch.no_grad():
        # Prefill
        from benchmark import prefill, forward_for_generate, sample
        next_token = prefill(model, encoded.view(1, -1), input_pos, temperature=0.0)
        generated_tokens = [next_token.item()]
        
        # Generate more tokens
        for i in range(20):  # Generate 20 more tokens
            input_pos = torch.tensor([encoded.size(0) + i], device=device, dtype=torch.int)
            logits = forward_for_generate(model, next_token.view(1, -1), input_pos)
            next_token, _ = sample(logits, temperature=0.0, top_k=1)
            generated_tokens.append(next_token.item())
    
    # Decode result
    full_tokens = tokenizer.encode(prompt) + generated_tokens
    result = tokenizer.decode(full_tokens)
    print(f"Generated: {result}")
    
    # Check for corruption indicators
    corruption_indicators = ['_REF', 'Silver', 'REF', 'silver']
    has_corruption = any(indicator in result for indicator in corruption_indicators)
    print(f"Has corruption indicators: {has_corruption}")
    
    print()


def step2_generation_settings_check(model_path, device):
    """Step 2: Make sure generation settings are strictly greedy"""
    print("=== Step 2: Generation Settings Check ===")
    
    # Load model and tokenizer
    model = load_model(Path(model_path), device, torch.bfloat16)
    tokenizer_path = Path(model_path).parent / "tokenizer.model"
    tokenizer = get_tokenizer(tokenizer_path, model_path)
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Two plus two equals"
    ]
    
    for prompt in test_prompts:
        print(f"\nTesting prompt: '{prompt}'")
        
        # Encode
        encoded = torch.tensor(tokenizer.encode(prompt), device=device, dtype=torch.int32)
        
        # Setup cache
        max_seq_len = len(encoded) + 64
        with torch.device(device):
            model.setup_caches(max_batch_size=1, max_seq_length=max_seq_len)
        
        # Generate with strict greedy settings
        input_pos = torch.arange(0, encoded.size(0), device=device)
        
        with torch.no_grad():
            from benchmark import prefill, forward_for_generate, sample
            
            # Prefill
            next_token = prefill(model, encoded.view(1, -1), input_pos, temperature=0.0)
            generated_tokens = [next_token.item()]
            
            # Generate more tokens with strict greedy settings
            for i in range(10):
                input_pos = torch.tensor([encoded.size(0) + i], device=device, dtype=torch.int)
                logits = forward_for_generate(model, next_token.view(1, -1), input_pos)
                next_token, _ = sample(logits, temperature=0.0, top_k=1)
                generated_tokens.append(next_token.item())
        
        # Decode and show first 100 characters
        full_tokens = tokenizer.encode(prompt) + generated_tokens
        result = tokenizer.decode(full_tokens)
        print(f"First 100 chars: {result[:100]}")
    
    print()


def step3_tokenizer_consistency_check(model_path, device):
    """Step 3: Check we're using the same tokenizer instance"""
    print("=== Step 3: Tokenizer Consistency Check ===")
    
    # Load tokenizer once
    tokenizer_path = model_path / "tokenizer.model"
    tokenizer = get_tokenizer(tokenizer_path, model_path)
    
    print(f"Tokenizer path: {tokenizer_path}")
    print(f"BOS id: {tokenizer.bos_id()}")
    print(f"EOS id: {tokenizer.eos_id()}")
    
    # Load model and check vocab size
    model = load_model(model_path, device, torch.bfloat16)
    print(f"Model vocab_size: {model.config.vocab_size}")
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
    
    # Test encode/decode consistency
    test_text = "Hello world test"
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)
    print(f"Test text: '{test_text}'")
    print(f"Encoded: {ids}")
    print(f"Decoded: '{decoded}'")
    print(f"Consistent: {test_text == decoded}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Comprehensive sanity check for LLM output corruption")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--step", type=int, choices=[0,1,2,3], help="Run specific step only")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print()
    
    if args.step is None or args.step == 0:
        save_system_info()
    
    if args.step is None or args.step == 1:
        step1_tokenizer_sanity_check(args.model, device)
        step1_model_sanity_check(args.model, device)
    
    if args.step is None or args.step == 2:
        step2_generation_settings_check(args.model, device)
    
    if args.step is None or args.step == 3:
        step3_tokenizer_consistency_check(args.model, device)
    
    print("Sanity check completed!")


if __name__ == "__main__":
    main() 