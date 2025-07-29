# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional
from safetensors.torch import load_file as load_safetensors_file
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf"),
    model_name: Optional[str] = None,
) -> None:
    print(f"Starting conversion for checkpoint_dir: {checkpoint_dir}")
    print(f"Model name: {model_name}")
    
    if model_name is None:
        model_name = checkpoint_dir.name
        print(f"Using checkpoint_dir.name as model_name: {model_name}")

    print(f"Loading model config for: {model_name}")
    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json_safetensors = checkpoint_dir / 'model.safetensors.index.json'
    model_map_json_pytorch = checkpoint_dir / "pytorch_model.bin.index.json"
    model_map_json = None
   
    print(f"Looking for safetensors index at: {model_map_json_safetensors}")
    try:
      assert model_map_json_safetensors.is_file()
      model_map_json = model_map_json_safetensors
      print(f"Found safetensors index at {model_map_json_safetensors}")
    except AssertionError:
      print(f"{model_map_json_safetensors} not found")
    if model_map_json is None:
      print(f"Looking for pytorch index at: {model_map_json_pytorch}")
      try:
        assert model_map_json_pytorch.is_file()
        model_map_json = model_map_json_pytorch
        print(f"Found pytorch index at {model_map_json_pytorch}")
      except AssertionError:
        print(f"{model_map_json_pytorch} not found")
   
    if model_map_json is None: 
        print("No model map found!")
        raise Exception("No model map found!")

    print(f"Loading model map from: {model_map_json}")
    with open(model_map_json) as json_map:
        bin_index = json.load(json_map)

    print(f"Found {len(bin_index['weight_map'])} weight mappings")
    print(f"Sample weight mappings: {list(bin_index['weight_map'].keys())[:5]}")

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        'model.layers.{}.self_attn.rotary_emb.inv_freq': None,
        'model.layers.{}.mlp.gate_proj.weight': 'layers.{}.feed_forward.w1.weight',
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    print(f"Found {len(bin_files)} bin files to process")

    def permute(w, n_head):
        dim = config.dim
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    print("Loading and merging state dicts...")
    merged_result = {}
    for i, file in enumerate(sorted(bin_files)):
       print(f"Processing file {i+1}/{len(bin_files)}: {file.name}")
       if "safetensors" in str(file):
           state_dict = load_safetensors_file(str(file), device="cpu")
           merged_result.update(state_dict)
       else:
           state_dict = torch.load(str(file), map_location="cpu", mmap=True, weights_only=True)
           merged_result.update(state_dict)
    
    print(f"Merged {len(merged_result)} tensors")
    print(f"Sample keys: {list(merged_result.keys())[:5]}")
    
    print("Converting weight mappings...")
    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    print(f"Converted to {len(final_result)} final tensors")
    print(f"Sample final keys: {list(final_result.keys())[:5]}")

    print("Processing attention weights...")
    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            
            # For Llama 3.1 with GQA, we need to handle the different head counts
            # Q has n_head heads, K and V have n_local_heads heads
            q = permute(q, config.n_head)
            k = permute(k, config.n_local_heads)
            v = permute(v, config.n_local_heads)
            
            # Concatenate Q, K, V into wqkv as expected by the model
            # The model expects total_head_dim = (n_head + 2 * n_local_heads) * head_dim
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
    
    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")
    print("Checkpoint saved successfully!")
    
    if 'llama-3' in model_name.lower():
        original_dir = checkpoint_dir / "original"
        tokenizer_model = original_dir / "tokenizer.model"
        tokenizer_model_tiktoken = checkpoint_dir / "tokenizer.model"
        print(f"Copying {tokenizer_model} to {tokenizer_model_tiktoken}")
        shutil.copy(tokenizer_model, tokenizer_model_tiktoken)
        print("Tokenizer copied successfully!")
    
    print("Conversion completed successfully!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"))
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
