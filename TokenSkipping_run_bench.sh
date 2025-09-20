#!/bin/bash

# ==== Config (edit as you like) ====
# export MODEL="Qwen/Qwen2-0.5B-Instruct"
export MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# export CTX="2048"
export CTX="32768"
export NEW="512"
export WINDOW="6144"

export ANCHORS="8"
export START_KEEP="0.75"
export END_KEEP="0.50"
export RESERVOIR="64"

# Use fa2 for performance; the 'attn' policy will override this to 'eager' as required.
export ATTN_IMPL="fa2"

# fast = counter-based only; exact = post-run scan; both = print both
export MEM_MEASURE="fast"

# Output CSV
export CSV="bench_fast.csv"

# Python exe (leave as python if your conda env is activated)
export PY="python"

echo
echo "===== Running BENCH suite on ${MODEL} (ctx=${CTX}, new=${NEW}) ====="
echo

# ---- Full KV ----
echo "[BASH] bench | method=Full KV (full) | attn_impl=${ATTN_IMPL}"
${PY} TokenSkipping.py bench --model "${MODEL}" --policy full --ctx ${CTX} --new ${NEW} --attn-impl ${ATTN_IMPL} --mem-measure ${MEM_MEASURE} --csv "${CSV}"

echo

# ---- Sliding Window (last-N) ----
echo "[BASH] bench | method=Sliding Window (last-N) (window) | window=${WINDOW} | attn_impl=${ATTN_IMPL}"
${PY} TokenSkipping.py bench --model "${MODEL}" --policy window --window ${WINDOW} --ctx ${CTX} --new ${NEW} --attn-impl ${ATTN_IMPL} --mem-measure ${MEM_MEASURE} --csv "${CSV}"

echo

# ---- Key-norm pruning ----
echo "[BASH] bench | method=Key-norm pruning (prune) | anchors=${ANCHORS} start=${START_KEEP} end=${END_KEEP} | attn_impl=${ATTN_IMPL}"
${PY} TokenSkipping.py bench --model "${MODEL}" --policy prune --anchors ${ANCHORS} --start-keep ${START_KEEP} --end-keep ${END_KEEP} --ctx ${CTX} --new ${NEW} --attn-impl ${ATTN_IMPL} --mem-measure ${MEM_MEASURE} --csv "${CSV}"

echo

# ---- TokenSkipping (quota) ----
echo "[BASH] bench | method=TokenSkipping (quota) (tskip) | anchors=${ANCHORS} reservoir=${RESERVOIR} start=${START_KEEP} end=${END_KEEP} | attn_impl=${ATTN_IMPL}"
${PY} TokenSkipping.py bench --model "${MODEL}" --policy tskip --anchors ${ANCHORS} --reservoir ${RESERVOIR} --start-keep ${START_KEEP} --end-keep ${END_KEEP} --ctx ${CTX} --new ${NEW} --attn-impl ${ATTN_IMPL} --mem-measure ${MEM_MEASURE} --csv "${CSV}"

echo

# ---- Attention-score pruning ----
echo "[BASH] bench | method=Attention-score pruning (attn) | anchors=${ANCHORS} start=${START_KEEP} end=${END_KEEP} (attn_impl=eager)"
${PY} TokenSkipping.py bench --model "${MODEL}" --policy attn --anchors ${ANCHORS} --start-keep ${START_KEEP} --end-keep ${END_KEEP} --ctx ${CTX} --new ${NEW} --attn-impl "eager" --mem-measure ${MEM_MEASURE} --csv "${CSV}"

echo
echo "===== BENCH suite complete. CSV -> ${CSV} ====="
echo