#!/bin/bash

# ==== Config (edit as you like) ====
# export MODEL="Qwen/Qwen2-0.5B-Instruct"
#export MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
export MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
export BLOCK="512"
export WINDOW="1024"

export ANCHORS="8"
export START_KEEP="0.75"
export END_KEEP="0.50"
export RESERVOIR="64"

# eager is fine here too
export ATTN_IMPL="eager"

# Choose one: PPL_FILE (if exists) OR PPL_DATASET (wikitext2|wikitext103|ptb)
export PPL_FILE=""
export PPL_DATASET="wikitext2"

# Output CSV
export CSV="bench_fast.csv"

# Python executable
export PY="python"

echo
echo "===== Running PPL suite on ${MODEL} (block=${BLOCK}) ====="
echo

# Decide input source
# This logic checks if PPL_FILE is set and exists, otherwise it falls back to PPL_DATASET.
if [[ -n "${PPL_FILE}" && -f "${PPL_FILE}" ]]; then
    SRC_FLAG="--ppl-file \"${PPL_FILE}\""
else
    SRC_FLAG="--ppl-dataset ${PPL_DATASET}"
fi

# ---- Full KV ----
echo "[BASH] ppl | method=Full KV (full)"
${PY} TokenSkipping.py ppl --model "${MODEL}" --policy full --block ${BLOCK} --attn-impl ${ATTN_IMPL} ${SRC_FLAG} --csv "${CSV}"

echo

# ---- Sliding Window (last-N) ----
echo "[BASH] ppl | method=Sliding Window (last-N) (window) | window=${WINDOW}"
${PY} TokenSkipping.py ppl --model "${MODEL}" --policy window --window ${WINDOW} --block ${BLOCK} --attn-impl ${ATTN_IMPL} ${SRC_FLAG} --csv "${CSV}"

echo

# ---- Key-norm pruning ----
echo "[BASH] ppl | method=Key-norm pruning (prune) | anchors=${ANCHORS} start=${START_KEEP} end=${END_KEEP}"
${PY} TokenSkipping.py ppl --model "${MODEL}" --policy prune --anchors ${ANCHORS} --start-keep ${START_KEEP} --end-keep ${END_KEEP} --block ${BLOCK} --attn-impl ${ATTN_IMPL} ${SRC_FLAG} --csv "${CSV}"

echo

# ---- TokenSkipping (quota) ----
echo "[BASH] ppl | method=TokenSkipping (quota) (tskip) | anchors=${ANCHORS} reservoir=${RESERVOIR} start=${START_KEEP} end=${END_KEEP}"
${PY} TokenSkipping.py ppl --model "${MODEL}" --policy tskip --anchors ${ANCHORS} --reservoir ${RESERVOIR} --start-keep ${START_KEEP} --end-keep ${END_KEEP} --block ${BLOCK} --attn-impl ${ATTN_IMPL} ${SRC_FLAG} --csv "${CSV}"

echo

# ---- Attention-score pruning ----
echo "[BASH] ppl | method=Attention-score pruning (attn) | anchors=${ANCHORS} start=${START_KEEP} end=${END_KEEP} (attn_impl=${ATTN_IMPL})"
${PY} TokenSkipping.py ppl --model "${MODEL}" --policy attn --anchors ${ANCHORS} --start-keep ${START_KEEP} --end-keep ${END_KEEP} --block ${BLOCK} --attn-impl ${ATTN_IMPL} ${SRC_FLAG} --csv "${CSV}"

echo
echo "===== PPL suite complete. CSV -> ${CSV} ====="
echo