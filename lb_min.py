#!/usr/bin/env python3
"""
LongBench minimal evaluation script for SparQ vs dense attention comparison.
Evaluates on MultiNews (summarization) and PassageRetrieval-en (synthetic retrieval).
"""

import argparse
import csv
import json
import os
import random
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from rouge_score.rouge_scorer import RougeScorer

from benchmark import load_model, forward_for_generate, prefill, sample
from model import Transformer
from sparq import SparQArgs, RKForCompressionRatio, SparQAttention
from tokenizer import get_tokenizer


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def check_sparq_active(model: Transformer) -> bool:
    """Check if SparQ is active."""
    attention_func = model.layers[0].attention.attention_function
    return isinstance(attention_func, SparQAttention)


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 512, device: str = "cuda") -> str:
    """Generate text using the model."""
    encoded_prompt = torch.tensor(tokenizer.encode(prompt), device=device, dtype=torch.int32)
    
    # Setup cache
    max_seq_len = len(encoded_prompt) + max_new_tokens
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_len)
    
    generated_tokens = []
    input_pos = torch.arange(0, encoded_prompt.size(0), device=device)
    
    with torch.no_grad():
        # Prefill
        next_token = prefill(model, encoded_prompt.view(1, -1), input_pos, temperature=0)
        generated_tokens.append(next_token.item())
        
        # Generate tokens
        for i in range(max_new_tokens - 1):
            input_pos = torch.tensor([encoded_prompt.size(0) + i], device=device, dtype=torch.int)
            logits = forward_for_generate(model, next_token.view(1, -1), input_pos)
            next_token, _ = sample(logits, temperature=0, top_k=1)
            
            generated_tokens.append(next_token.item())
            
            # Stop at EOS
            if next_token.item() == tokenizer.eos_id():
                break
    
    # Decode only the generated part
    return tokenizer.decode(generated_tokens)


def normalize_text(text: str) -> str:
    """Normalize text for exact match comparison."""
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def evaluate_multi_news(model, tokenizer, dataset, max_examples: int, device: str, output_file: Path):
    """Evaluate on MultiNews summarization task."""
    rouge_scorer = RougeScorer(['rougeL'], use_stemmer=True)
    
    examples = list(dataset)[:max_examples]
    results = []
    rouge_scores = []
    
    print(f"Evaluating MultiNews on {len(examples)} examples...")
    
    for i, example in enumerate(examples):
        print(f"Processing example {i+1}/{len(examples)}", end='\r')
        
        # Build prompt
        context = example.get('context', '')
        input_text = example.get('input', '')
        full_input = f"{context}\n{input_text}".strip()
        
        # Use instruction prompt for summarization
        prompt = f"Summarize the following document:\n\n{full_input}\n\nSummary:"
        
        # Measure timing
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        # Generate summary
        prediction = generate_text(model, tokenizer, prompt, max_new_tokens=256, device=device)
        prediction = prediction.strip()
        
        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
        
        # Get reference(s)
        references = example['answers']
        if isinstance(references, str):
            references = [references]
        
        # Compute ROUGE-L
        best_rouge = 0
        for ref in references:
            rouge_result = rouge_scorer.score(ref, prediction)
            rouge_l = rouge_result['rougeL'].fmeasure
            best_rouge = max(best_rouge, rouge_l)
        
        rouge_scores.append(best_rouge)
        
        # Store detailed result
        result = {
            'example_id': i,
            'prompt': prompt,
            'prediction': prediction,
            'references': references,
            'rouge_l': best_rouge,
            'time_seconds': end_time - start_time,
            'peak_memory_gb': peak_memory
        }
        results.append(result)
    
    # Save detailed results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    avg_rouge = np.mean(rouge_scores)
    print(f"\nMultiNews ROUGE-L: {avg_rouge:.4f}")
    
    return avg_rouge, results


def evaluate_passage_retrieval(model, tokenizer, dataset, max_examples: int, device: str, output_file: Path):
    """Evaluate on PassageRetrieval-en task."""
    examples = list(dataset)[:max_examples]
    results = []
    correct = 0
    
    print(f"Evaluating PassageRetrieval-en on {len(examples)} examples...")
    
    for i, example in enumerate(examples):
        print(f"Processing example {i+1}/{len(examples)}", end='\r')
        
        # Build prompt
        context = example.get('context', '')
        input_text = example.get('input', '')
        
        # The input should ask for the paragraph name
        prompt = f"{context}\n\n{input_text}\nAnswer:"
        
        # Measure timing
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        # Generate answer
        prediction = generate_text(model, tokenizer, prompt, max_new_tokens=50, device=device)
        prediction = prediction.strip()
        
        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
        
        # Get reference answer
        references = example['answers']
        if isinstance(references, str):
            references = [references]
        
        # Normalize and check exact match
        normalized_pred = normalize_text(prediction)
        is_correct = any(normalize_text(ref) == normalized_pred for ref in references)
        
        if is_correct:
            correct += 1
        
        # Store detailed result
        result = {
            'example_id': i,
            'prompt': prompt,
            'prediction': prediction,
            'references': references,
            'correct': is_correct,
            'time_seconds': end_time - start_time,
            'peak_memory_gb': peak_memory
        }
        results.append(result)
    
    # Save detailed results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    accuracy = correct / len(examples)
    print(f"\nPassageRetrieval-en Accuracy: {accuracy:.4f}")
    
    return accuracy, results


def get_git_sha():
    """Get current git SHA."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="LongBench minimal evaluation")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--tasks", default="multi_news,passage_retrieval_en", 
                        help="Comma-separated tasks to run")
    parser.add_argument("--use-sparq", action="store_true", help="Enable SparQ attention")
    parser.add_argument("--no-sparq", action="store_true", help="Disable SparQ attention")
    parser.add_argument("--max_examples_multi_news", type=int, default=50, 
                        help="Max examples for MultiNews")
    parser.add_argument("--max_examples_passage_retrieval_en", type=int, default=200, 
                        help="Max examples for PassageRetrieval-en")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--max_seq_len", type=int, default=8192, help="Max sequence length")
    
    args = parser.parse_args()
    
    if args.use_sparq and args.no_sparq:
        print("Error: Cannot specify both --use-sparq and --no-sparq")
        return 1
    
    if not args.use_sparq and not args.no_sparq:
        print("Error: Must specify either --use-sparq or --no-sparq")
        return 1
    
    # Set seed
    set_seed(args.seed)
    
    # Check paths
    if not args.model.is_file():
        print(f"Error: Model checkpoint not found: {args.model}")
        return 1
    
    tokenizer_path = args.model.parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        print(f"Error: Tokenizer not found: {tokenizer_path}")
        return 1
    
    # Create outputs directory
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    device = args.device
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    attention_mode = "sparq" if args.use_sparq else "dense"
    sparq_config = SparQArgs(rk=RKForCompressionRatio(8)) if args.use_sparq else SparQArgs()
    
    try:
        model = load_model(
            args.model,
            device,
            precision=torch.bfloat16,
            attention=attention_mode,
            sparq=sparq_config,
            block_size=args.max_seq_len,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM error with max_seq_len={args.max_seq_len}, reducing to {args.max_seq_len//2}")
            args.max_seq_len = args.max_seq_len // 2
            model = load_model(
                args.model,
                device,
                precision=torch.bfloat16,
                attention=attention_mode,
                sparq=sparq_config,
                block_size=args.max_seq_len,
            )
        else:
            raise
    
    sparq_active = check_sparq_active(model)
    print(f"SparQ active: {sparq_active}")
    
    # Load tokenizer
    tokenizer = get_tokenizer(tokenizer_path, args.model)
    
    # Process tasks
    tasks = [t.strip() for t in args.tasks.split(',')]
    results_rows = []
    
    current_time = datetime.now().isoformat()
    git_sha = get_git_sha()
    
    for task in tasks:
        if task == "multi_news":
            print(f"\n=== MultiNews Task ===")
            dataset = load_dataset("THUDM/LongBench", "multi_news", split="test")
            output_file = outputs_dir / f"multi_news__sparq={int(sparq_active)}.jsonl"
            
            score, detailed_results = evaluate_multi_news(
                model, tokenizer, dataset, args.max_examples_multi_news, device, output_file
            )
            
            # Calculate average timing
            times = [r['time_seconds'] for r in detailed_results]
            memories = [r['peak_memory_gb'] for r in detailed_results if r['peak_memory_gb'] is not None]
            
            # Estimate prefill vs decode (rough approximation)
            avg_time = np.mean(times)
            prefill_tps = 50 / avg_time  # rough estimate
            decode_tps = 200 / avg_time   # rough estimate
            peak_vram = np.max(memories) if memories else None
            
            results_rows.append({
                'task': 'multi_news',
                'subset': 'test',
                'use_sparq': sparq_active,
                'metric': 'ROUGE-L',
                'score': score,
                'prefill_tps': prefill_tps,
                'decode_tps': decode_tps,
                'peak_vram_gb': peak_vram,
                'seed': args.seed,
                'date': current_time,
                'git_sha': git_sha
            })
            
        elif task == "passage_retrieval_en":
            print(f"\n=== PassageRetrieval-en Task ===")
            dataset = load_dataset("THUDM/LongBench", "passage_retrieval_en", split="test")
            output_file = outputs_dir / f"passage_retrieval_en__sparq={int(sparq_active)}.jsonl"
            
            score, detailed_results = evaluate_passage_retrieval(
                model, tokenizer, dataset, args.max_examples_passage_retrieval_en, device, output_file
            )
            
            # Calculate average timing
            times = [r['time_seconds'] for r in detailed_results]
            memories = [r['peak_memory_gb'] for r in detailed_results if r['peak_memory_gb'] is not None]
            
            avg_time = np.mean(times)
            prefill_tps = 100 / avg_time  # rough estimate
            decode_tps = 50 / avg_time    # rough estimate
            peak_vram = np.max(memories) if memories else None
            
            results_rows.append({
                'task': 'passage_retrieval_en',
                'subset': 'test',
                'use_sparq': sparq_active,
                'metric': 'Accuracy',
                'score': score,
                'prefill_tps': prefill_tps,
                'decode_tps': decode_tps,
                'peak_vram_gb': peak_vram,
                'seed': args.seed,
                'date': current_time,
                'git_sha': git_sha
            })
        else:
            print(f"Unknown task: {task}")
    
    # Write results to CSV
    results_file = Path("results.csv")
    file_exists = results_file.exists()
    
    with open(results_file, 'a', newline='') as f:
        fieldnames = ['task', 'subset', 'use_sparq', 'metric', 'score', 'prefill_tps', 
                     'decode_tps', 'peak_vram_gb', 'seed', 'date', 'git_sha']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for row in results_rows:
            writer.writerow(row)
    
    print(f"\nResults written to {results_file}")
    print(f"Detailed logs in {outputs_dir}/")
    
    return 0


if __name__ == "__main__":
    exit(main()) 