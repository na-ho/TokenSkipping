# TokenSkipping — Practical KV-Cache Pruning for Long-Context LLM Inference

Lightweight **benchmark + PPL** harness for **KV-cache pruning/retention** in LLM inference.  
**Main:** `TokenSkipping.py` · **Runners:** `TokenSkipping_run_bench.sh`, `TokenSkipping_run_ppl.sh` · **Extras:** `TokenSkipping_benchmark_suite.py`

## Paper support
This repo includes the **TokenSkipping (tskip)** policy and is intended to support:  
**“TokenSkipping: A Practical and Robust KV Cache Pruning Method for Long-Context LLM Inference.”**  
## Install
Tested inside: **`pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`**

```bash
pip install transformers
pip install accelerate
pip install datasets
pip install flash-attn --no-build-isolation
````

## Quick start

### Bench (throughput & KV memory)

```bash
python TokenSkipping.py bench \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --policy tskip \
  --ctx 32768 --new 512 \
  --anchors 8 --reservoir 64 \
  --start-keep 0.75 --end-keep 0.50 \
  --attn-impl fa2 \
  --mem-measure fast \
  --csv bench.csv
# or: bash TokenSkipping_run_bench.sh
```

### PPL

```bash
# Dataset
python TokenSkipping.py ppl \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --policy window --window 1024 \
  --block 512 --ppl-dataset wikitext2 --csv ppl.csv

# Custom text
python TokenSkipping.py ppl \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --policy tskip --block 512 \
  --ppl-file /path/to/text.txt --csv ppl_custom.csv
# or: bash TokenSkipping_run_ppl.sh
```

### Suite (ratio sweep)

```bash
python TokenSkipping_benchmark_suite.py chart \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --policy tskip --block 512 \
  --ppl-dataset wikitext2 --ppl-ratios 1.0 0.75 0.5 0.25 \
  --csv chart_sweep.csv
```

## Policies

`full`, `window`/`fifo`, `prune` (L2-norm), `attn`, `tskip` (TokenSkipping)


## Citation
If you use this code or method, please cite our paper:

**TokenSkipping: A Practical and Robust KV Cache Pruning Method for Long-Context LLM Inference**  
**Authors:**
Narupol Hongthai
Ekapol Chuangsuwanich
*(Chulalongkorn University)*

## Updates
For the latest version of this code and issue tracking, please visit:  
https://github.com/na-ho/TokenSkipping
