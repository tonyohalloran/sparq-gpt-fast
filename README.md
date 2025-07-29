# gpt-fast with SparQ Attention
We extend [gpt-fast](https://github.com/pytorch-labs/gpt-fast) to support SparQ attention, a bandwidth-efficient attention algorithm that speeds up generation for existing LLMs with no fine tuning.
For details of SparQ, see [the paper](https://arxiv.org/pdf/2312.04985).

The `main` branch tracks the gpt-fast repo. The `with-sparq` branch contains our modifications. You can [compare "main" and "with-sparq"](https://github.com/graphcore-research/sparq-gpt-fast/compare/main...with-sparq) to see what we added.

You might also be interested in [sparq-llama.cpp](https://github.com/graphcore-research/sparq-llama.cpp), our implementation of SparQ in llama.cpp.


## Results
We obtain the following speedups on an H100 PCIe, using BF16 for the model parameters and KV cache, and compressing the memory transfers 8x with SparQ:
![Plot showing benchmark results](example_benchmark.png)
"estimated theoretical max" shows an estimate of the best-cast speedup that could be achieved by SparQ if the attention operation was purely memory-bound, and all compute and communication was overlapped. See `theoretical_speedups.py` for how this is calculated.


## How to reproduce the results
1. Install Python >=3.10
2. Install the requirements: `pip install -r requirements.txt`
3. Run `huggingface-cli login` or set the `HF_TOKEN` environment variable. The associated account must have access to `meta-llama/Llama-2-7b-chat-hf`
3. Download Llama 2 7b from Hugging Face, and prepare it for gpt-fast: `./scripts/prepare.sh "meta-llama/Llama-2-7b-chat-hf"`
4. Updated `expected_gpu` in `run_speedup_benchmark.py` to the expected model of GPU (this avoid accidentally comparing results from different GPUs)
5. Run the benchmark: `python run_speedup_benchmark.py`

SparQ is implemented in PyTorch, not as a custom kernel.
However, we found that torch.compile() was able to generate a performant implementation.

---

# 🧋 **Hey Friend! LongBench SparQ Evaluation (You Get Boba For This!)**

Thanks for running this on your pod! 🙏 I owe you boba after this evaluation finishes successfully. Here are the **super simple** steps:

## 🚀 **Step 1: Clone & Environment Setup (5 minutes)**

```bash
# Clone Tony's repo
git clone https://github.com/tonyohalloran/sparq-gpt-fast.git
cd sparq-gpt-fast
git checkout with-sparq

# Make virtual environment  
python3 -m venv .venv
source .venv/bin/activate

# Install everything
pip install --upgrade pip
pip install -r requirements.txt
pip install datasets rouge-score evaluate
```

## 🤗 **Step 2: Get Hugging Face Access (2 minutes)**

You need access to Llama-3.1-8B-Instruct:

```bash
# Either login interactively:
huggingface-cli login

# OR set token directly (get from https://huggingface.co/settings/tokens):
export HF_TOKEN="your_token_here"
```

## 📦 **Step 3: Download & Prepare Model (10-15 minutes)**

```bash
# This downloads and converts Llama-3.1-8B-Instruct for gpt-fast
./scripts/prepare.sh "meta-llama/Llama-3.1-8B-Instruct"
```

**⚠️ Expected:** This will download ~16GB and take 10-15 minutes. You should see files created in `checkpoints/Llama-3.1-8B-Instruct/`.

## ✅ **Step 4: Sanity Check (30 seconds)**

Make sure everything works:

```bash
# Test SparQ ON
python sanity_check.py --model ./checkpoints/Llama-3.1-8B-Instruct --use-sparq

# Test SparQ OFF  
python sanity_check.py --model ./checkpoints/Llama-3.1-8B-Instruct --no-sparq
```

**✅ Expected output:** You should see `SparQ active: True` and `SparQ active: False` respectively, plus some generated text.

## 🏃‍♀️ **Step 5: Run The Main Evaluation (30-60 minutes)**

Now the real deal! Run these **4 commands in order**:

```bash
# Baseline (no SparQ)
python lb_min.py --model ./checkpoints/Llama-3.1-8B-Instruct --tasks multi_news,passage_retrieval_en --no-sparq

# SparQ version  
python lb_min.py --model ./checkpoints/Llama-3.1-8B-Instruct --tasks multi_news,passage_retrieval_en --use-sparq
```

**⏱️ Timeline:**
- MultiNews (50 examples): ~15-20 minutes each run
- PassageRetrieval-en (200 examples): ~10-15 minutes each run  
- **Total: ~50-70 minutes for both runs**

## 📊 **Step 6: Send Results Back (1 minute)**

When done, you'll have:
- `results.csv` - The money shot with all the numbers
- `outputs/` folder - Detailed logs

Send Tony these files! 🎉

## 🆘 **If Something Breaks:**

### **"CUDA out of memory"**
The script auto-handles this, but if it keeps failing:
```bash
python lb_min.py --model ./checkpoints/Llama-3.1-8B-Instruct --tasks multi_news,passage_retrieval_en --no-sparq --max_seq_len 4096
```

### **"Model not found"** 
Check that `./checkpoints/Llama-3.1-8B-Instruct/model.pth` exists after Step 3.

### **"Permission denied for Llama"**
Make sure your HuggingFace account has access to meta-llama models.

### **Other issues**
Text Tony! The scripts have good error messages.

## 🧋 **Boba Redemption:**
After you send back `results.csv` with 4 rows (2 tasks × SparQ on/off), Tony owes you boba of your choice! 

**Expected final `results.csv`:**
```csv
task,subset,use_sparq,metric,score,prefill_tps,decode_tps,peak_vram_gb,seed,date,git_sha
multi_news,test,False,ROUGE-L,0.XXX,XX.X,XXX.X,X.X,123,YYYY-MM-DD...,abc123
multi_news,test,True,ROUGE-L,0.XXX,XX.X,XXX.X,X.X,123,YYYY-MM-DD...,abc123  
passage_retrieval_en,test,False,Accuracy,0.XXX,XX.X,XXX.X,X.X,123,YYYY-MM-DD...,abc123
passage_retrieval_en,test,True,Accuracy,0.XXX,XX.X,XXX.X,X.X,123,YYYY-MM-DD...,abc123
```

You're the best! 🚀

---

## Original LongBench Evaluation Details

### Technical Notes
- Data loaded from **THUDM/LongBench** only; no locally fabricated datasets
- **MultiNews** (summarization): ROUGE-L metric, 50 examples  
- **PassageRetrieval-en** (synthetic retrieval): Accuracy (exact match), 200 examples
- Requires NVIDIA GPU with ~10GB+ VRAM
- Uses deterministic generation (temperature=0) for reproducibility
- Auto-handles OOM by reducing sequence length

---

## License
This repo is based off [gpt-fast](https://github.com/pytorch-labs/gpt-fast), which is released under the BSD 3 license.
We also release our modifications under the BSD 3 license.
See [LICENSE](LICENSE).

