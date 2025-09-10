#!/usr/bin/env python3
from __future__ import annotations

import os, math, time, argparse, warnings, csv, datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Iterable

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

from huggingface_hub import login
login(token="")

logger = logging.get_logger(__name__)

# ============================== small utils ==============================

def _device_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _ensure_pad(cfg):
    if getattr(cfg, "pad_token_id", None) is None and getattr(cfg, "eos_token_id", None) is not None:
        cfg.pad_token_id = cfg.eos_token_id

def linear_schedule(start: float, end: float, num_layers: int) -> List[float]:
    if num_layers <= 1:
        return [start]
    return [float(start + (end - start) * (i / (num_layers - 1))) for i in range(num_layers)]

def _kv_bytes_per_token_per_layer(model) -> int:
    dtype = next(model.parameters()).dtype
    itemsize = 4
    if dtype == torch.float16 or dtype == torch.bfloat16: itemsize = 2
    n_heads = getattr(model.config, "num_attention_heads")
    n_kv = getattr(model.config, "num_key_value_heads", n_heads)
    head_dim = model.config.hidden_size // n_heads
    return 2 * itemsize * n_kv * head_dim  # K and V

def _tensor_nbytes(x: Optional[torch.Tensor]) -> int:
    if x is None: return 0
    try:
        return x.element_size() * x.numel()
    except Exception:
        return 0

POLICY_LABELS = {
    "full":  "Full KV",
    "window":"Sliding Window (last-N)",
    "fifo":  "Sliding Window (last-N)",
    "prune": "Key-norm pruning",
    "attn":  "Attention-score pruning",
    "tskip": "TokenSkipping (quota)",
}

def policy_label(s: str) -> str:
    return POLICY_LABELS.get(s, s)

# ============================== Base cache (speed-neutral counters) ==============================
class BasePolicyCache(DynamicCache):
    """A DynamicCache wrapper that keeps *lightweight* counters to estimate KV usage
       without scanning tensors during decode. We update counts only when we already
       touch Python (on cache update or when we explicitly slice after a policy prune).

       - cur_len[i]: number of tokens currently stored at layer i
       - peak_len[i]: max of cur_len[i] over the whole run (decode-peak)
       These let us estimate KV bytes precisely: len * (2 * heads_kv * head_dim * itemsize).

       We also expose optional exact end-current bytes via a single scan *after* timing.
    """
    def __init__(self, num_layers: int, num_heads: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seen = [0 for _ in range(num_layers)]
        self.kept = [0 for _ in range(num_layers)]
        self.tokens_so_far = [0 for _ in range(num_layers)]
        # speed-neutral length counters
        self.cur_len = [0 for _ in range(num_layers)]
        self.peak_len = [0 for _ in range(num_layers)]
        # debug (optional)
        self._dbg_last: List[Optional[Dict[str, Any]]] = [None] * num_layers

    # ---- compatibility helpers for exact (post-run) reads ----
    def _get_kv(self, i: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        try:
            layer = self.layers[i]  # HF>=4.56
            return getattr(layer, "keys", None), getattr(layer, "values", None)
        except Exception:
            pass
        try:
            return self.key_cache[i], self.value_cache[i]  # legacy
        except Exception:
            return None, None

    # ---- counter updates ----
    def _inc_cur_len(self, i: int, appended: int):
        self.cur_len[i] += int(appended)
        if self.cur_len[i] > self.peak_len[i]:
            self.peak_len[i] = self.cur_len[i]

    def _set_cur_len(self, i: int, new_len: int):
        self.cur_len[i] = int(new_len)
        if self.cur_len[i] > self.peak_len[i]:
            self.peak_len[i] = self.cur_len[i]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # key_states has already been pruned by the policy for this step
        S = int(key_states.shape[2])
        out = super().update(key_states, value_states, layer_idx, cache_kwargs=cache_kwargs)
        # appending S tokens to this layer
        self._inc_cur_len(layer_idx, S)
        return out

    # --- eviction helpers (must also update counters) ---
    def _slice_cache_lastN(self, layer_idx: int, N: int):
        # no need to read tensors; just clamp length counters
        new_len = min(self.cur_len[layer_idx], int(N))
        self._set_cur_len(layer_idx, new_len)
        # best-effort real slicing to avoid growth beyond N (outside timing-critical region)
        try:
            K, V = self._get_kv(layer_idx)
            if K is not None and K.shape[2] > N:
                layer = self.layers[layer_idx]
                layer.keys = K[:, :, -N:, :]
                layer.values = V[:, :, -N:, :]
        except Exception:
            pass

    def _slice_cache_mask(self, layer_idx: int, keep_mask_1d: torch.Tensor):
        """keep_mask_1d: [T] boolean over sequence dimension."""
        new_len = int(keep_mask_1d.sum().item())
        self._set_cur_len(layer_idx, new_len)
        # best-effort real slicing (post-timing or eager-only sections)
        try:
            K, V = self._get_kv(layer_idx)
            if K is None: return
            km = keep_mask_1d.to(K.device)
            if km.ndim != 1: km = km.flatten()
            if km.numel() != K.shape[2]:
                return
            layer = self.layers[layer_idx]
            layer.keys = K[:, :, km, :]
            layer.values = V[:, :, km, :]
        except Exception:
            pass

    # ---- reporting ----
    def retention_stats(self) -> List[Tuple[int,int,float]]:
        out = []
        for s,k in zip(self.seen, self.kept):
            out.append((s, k, (k / max(1, s))))
        return out

    def kv_estimated_bytes(self, bytes_per_tok_per_layer: int) -> Tuple[int,int]:
        cur = sum(self.cur_len) * bytes_per_tok_per_layer
        peak = sum(self.peak_len) * bytes_per_tok_per_layer
        return cur, peak

    def kv_exact_end_bytes(self) -> int:
        total = 0
        # Try new API first
        try:
            for lyr in self.layers:
                total += _tensor_nbytes(getattr(lyr, "keys", None))
                total += _tensor_nbytes(getattr(lyr, "values", None))
            return total
        except Exception:
            pass
        # Fallback to legacy
        try:
            for i in range(len(self.key_cache)):
                total += _tensor_nbytes(self.key_cache[i])
                total += _tensor_nbytes(self.value_cache[i])
        except Exception:
            pass
        return total

# ============================== policies ==============================
@dataclass
class PolicyArgs:
    anchors: int = 8
    reservoir: int = 64
    start_keep: float = 1.0
    end_keep: float = 1.0
    debug_keep: bool = False
    stream_exact: bool = False
    window: int = 1024

# Full (no skipping)
class FullCache(BasePolicyCache):
    def __init__(self, L, H, args: PolicyArgs):
        super().__init__(L, H)
        self.args = args

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        ks = key_states
        B,H,S,D = ks.shape
        kept_now = int(B*S)
        self.seen[layer_idx] += int(B*S)
        self.kept[layer_idx] += kept_now
        self.tokens_so_far[layer_idx] += int(S)
        return super().update(key_states, value_states, layer_idx, cache_kwargs=cache_kwargs)

# Sliding window (last-N)
# Sliding window (last-N)
class WindowCache(BasePolicyCache):
    def __init__(self, L, H, args: PolicyArgs):
        super().__init__(L, H)
        self.args = args
        self.window = int(args.window)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        S = key_states.shape[2]

        # Update statistics
        self.seen[layer_idx] += S
        self.kept[layer_idx] += S # All new tokens are initially kept
        self.tokens_so_far[layer_idx] += S

        # ========================= FIX STARTS HERE =========================
        # The original code's counter logic was flawed for this policy.
        # This new implementation correctly manages the length counters for a sliding window.

        # 1. Update the actual tensors in the cache by calling the grandparent's method.
        #    This bypasses the BasePolicyCache.update() which sets the wrong peak length.
        out = DynamicCache.update(self, key_states, value_states, layer_idx, cache_kwargs)

        # 2. After adding S tokens, determine the new length, capped by the window size.
        #    We use the previous cur_len to calculate this.
        new_len = min(self.cur_len[layer_idx] + S, self.window)

        # 3. Update both cur_len and peak_len with this correct, window-constrained length.
        #    This is the key: peak_len is now correctly capped by the window size.
        self.cur_len[layer_idx] = new_len
        if new_len > self.peak_len[layer_idx]:
            self.peak_len[layer_idx] = new_len

        # 4. Physically slice the cache tensors to enforce the window.
        #    This ensures the actual tensors don't grow beyond the window size.
        self._slice_cache_lastN(layer_idx, self.window)
        # ========================== FIX ENDS HERE ==========================

        return out

# Key-norm pruning (prefill top-k per update; anchors kept)
class PruneNormCache(BasePolicyCache):
    def __init__(self, L, H, args: PolicyArgs):
        super().__init__(L, H)
        self.args = args
        self.schedule = linear_schedule(args.start_keep, args.end_keep, L)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        ks, vs = key_states, value_states
        B,H,S,D = ks.shape
        with torch.no_grad():
            norms = torch.linalg.vector_norm(ks, dim=-1).mean(dim=1)  # [B,S]
        keep_frac = max(0.0, min(1.0, self.schedule[layer_idx]))
        start_pos = self.tokens_so_far[layer_idx]
        end_pos = start_pos + S
        self.tokens_so_far[layer_idx] = end_pos
        anchors_global = int(self.args.anchors)
        pos = torch.arange(start_pos, end_pos, device=ks.device)
        anchor_vec = (pos < anchors_global)  # [S]

        target_keep = int(round(keep_frac * S))
        anchors_in_update = int(anchor_vec.sum().item())
        budget_non = max(0, target_keep - anchors_in_update)

        if budget_non > 0 and S > anchors_in_update:
            idx_all = torch.arange(S, device=ks.device)
            idx_non = idx_all[~anchor_vec]
            norms_non = norms[0, ~anchor_vec]
            k_non = min(budget_non, idx_non.numel())
            topk_idx = torch.topk(norms_non, k_non, largest=True, sorted=False).indices
            keep_idx = torch.cat([idx_all[anchor_vec], idx_non[topk_idx]], dim=0)
        else:
            keep_idx = torch.arange(S, device=ks.device)[anchor_vec]

        keep_mask = torch.zeros((S,), dtype=torch.bool, device=ks.device)
        keep_mask[keep_idx] = True
        keep_mask = keep_mask.unsqueeze(0).expand(B, S)

        kept_now = int(keep_mask.sum().item())
        self.seen[layer_idx] += int(B*S)
        self.kept[layer_idx] += kept_now

        ks_out = ks[:, :, keep_mask[0], :]
        vs_out = vs[:, :, keep_mask[0], :]
        return super().update(ks_out, vs_out, layer_idx, cache_kwargs=cache_kwargs)

# TokenSkipping (quota) baseline
class Reservoir:
    def __init__(self, k: int):
        self.k = k
        self.buf = []
        self.n_seen = 0
    def add(self, x: torch.Tensor):
        x = x.detach().float().flatten()
        for v in x.tolist():
            self.n_seen += 1
            if len(self.buf) < self.k:
                self.buf.append(v)
            else:
                j = torch.randint(0, self.n_seen, (1,)).item()
                if j < self.k:
                    self.buf[j] = v
    def percentile(self, p: float) -> float:
        if not self.buf: return float("inf")
        b = sorted(self.buf)
        idx = min(len(b)-1, max(0, int(p * (len(b)-1))))
        return float(b[idx])

class TokenSkippingCache(BasePolicyCache):
    def __init__(self, L, H, args: PolicyArgs):
        super().__init__(L, H)
        self.args = args
        self.schedule = linear_schedule(args.start_keep, args.end_keep, L)
        self.reservoir = [Reservoir(args.reservoir) for _ in range(L)]
        self.kept_non = [0 for _ in range(L)]
        self.debug_keep = args.debug_keep

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        ks, vs = key_states, value_states
        B,H,S,D = ks.shape
        with torch.no_grad():
            norms_h = torch.linalg.vector_norm(ks, dim=-1)  # [B,H,S]
            norms = norms_h.mean(dim=1)  # [B,S]

        keep_frac = max(0.0, min(1.0, self.schedule[layer_idx]))
        start_pos = self.tokens_so_far[layer_idx]
        end_pos = start_pos + S
        self.tokens_so_far[layer_idx] = end_pos
        anchors_global = int(self.args.anchors)
        pos = torch.arange(start_pos, end_pos, device=ks.device)
        anchor_vec = (pos < anchors_global)  # [S]

        if S > 1:
            target_keep = int(round(keep_frac * S))
            anchors_in_update = int(anchor_vec.sum().item())
            budget_non = max(0, target_keep - anchors_in_update)
            if budget_non > 0 and S > anchors_in_update:
                idx_all = torch.arange(S, device=ks.device)
                idx_non = idx_all[~anchor_vec]
                norms_non = norms[0, ~anchor_vec]
                k_non = min(budget_non, idx_non.numel())
                topk_idx = torch.topk(norms_non, k_non, largest=True, sorted=False).indices
                keep_idx = torch.cat([idx_all[anchor_vec], idx_non[topk_idx]], dim=0)
            else:
                keep_idx = torch.arange(S, device=ks.device)[anchor_vec]
            keep_mask = torch.zeros((S,), dtype=torch.bool, device=ks.device)
            keep_mask[keep_idx] = True
            keep_mask = keep_mask.unsqueeze(0).expand(B, S)
            kept_now = int(keep_mask.sum().item())
            self.seen[layer_idx] += int(B*S)
            self.kept[layer_idx] += kept_now
            kept_non_now = int((keep_mask & (~anchor_vec.unsqueeze(0))).sum().item())
            self.kept_non[layer_idx] += kept_non_now
            mode = "topk"
        else:
            # streaming S==1: percentile + strict quota on non-anchors
            q = 1.0 - keep_frac
            if self.reservoir[layer_idx].n_seen < max(8, 32):
                thr = float("-inf") if anchor_vec.item() else norms[0,0].item()
            else:
                thr = self.reservoir[layer_idx].percentile(q)
            is_anchor = bool(anchor_vec.item())
            if is_anchor:
                keep_flag = True
            else:
                non_anchor_pos = max(0, end_pos - anchors_global)
                target_kept_non = int(round(keep_frac * non_anchor_pos))
                under_budget = self.kept_non[layer_idx] < target_kept_non
                keep_flag = under_budget and (norms[0,0].item() >= thr)
            keep_mask = anchor_vec.unsqueeze(0) if is_anchor else torch.tensor([[keep_flag]], device=ks.device)
            kept_now = int(keep_mask.sum().item())
            self.seen[layer_idx] += int(B*S)
            self.kept[layer_idx] += kept_now
            if (not is_anchor) and keep_flag:
                self.kept_non[layer_idx] += 1
            mode = "stream"

        # reservoir update
        self.reservoir[layer_idx].add(norms[0])
        ks_out = ks[:, :, keep_mask[0], :]
        vs_out = vs[:, :, keep_mask[0], :]
        if self.debug_keep:
            self._dbg_last[layer_idx] = {
                "S": int(S), "mode": mode, "range": f"[{start_pos},{end_pos})",
                "keep_frac": float(keep_frac),
                "anchors_global": int(anchors_global),
                "kept_now": int(kept_now),
            }
        return super().update(ks_out, vs_out, layer_idx, cache_kwargs=cache_kwargs)

# Attention-score pruning baseline (needs eager for attentions)
class AttnPruneCache(BasePolicyCache):
    def __init__(self, L, H, args: PolicyArgs):
        super().__init__(L, H)
        self.args = args
        self.schedule = linear_schedule(args.start_keep, args.end_keep, L)
        self.anchors = int(args.anchors)
        self._warned_no_attn = False

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        ks, vs = key_states, value_states
        B,H,S,D = ks.shape

        # This method intentionally passes all tokens through, as pruning is handled
        # post-attention in the post_decode_prune method.
        ks_out = ks
        vs_out = vs
        kept_now = int(B * S)

        self.seen[layer_idx] += int(B*S)
        self.kept[layer_idx] += kept_now
        self.tokens_so_far[layer_idx] += int(S)

        return super().update(ks_out, vs_out, layer_idx, cache_kwargs=cache_kwargs)

    def post_decode_prune(self, attentions: Optional[List[torch.Tensor]]):
        if attentions is None:
            if not self._warned_no_attn:
                warnings.warn(
                    "output_attentions=None; cannot run attention pruning. "
                    "Ensure --attn-impl eager. Continuing without pruning for this step."
                )
                self._warned_no_attn = True
            return
        for i, attn in enumerate(attentions):
            if attn is None:
                continue
            keep_frac = max(0.0, min(1.0, self.schedule[i]))
            T = attn.shape[-1]
            anchors_seen = min(self.anchors, T)
            target_keep = int(round(keep_frac * T))
            budget_non = max(0, target_keep - anchors_seen)
            scores = attn.mean(dim=(0,1)).squeeze(0)  # [T]
            idx_all = torch.arange(T, device=scores.device)
            idx_non = idx_all[anchors_seen:]
            if budget_non <= 0:
                keep_mask = torch.zeros((T,), dtype=torch.bool, device=scores.device)
                keep_mask[:anchors_seen] = True
            else:
                k_non = min(budget_non, idx_non.numel())
                keep_mask = torch.zeros((T,), dtype=torch.bool, device=scores.device)
                keep_mask[:anchors_seen] = True
                if k_non > 0 and idx_non.numel() > 0:
                    topk_idx = torch.topk(scores[idx_non], k_non, largest=True, sorted=False).indices
                    choose = idx_non[topk_idx]
                    keep_mask[choose] = True
            # apply to cache & counters
            self._slice_cache_mask(i, keep_mask)

# ============================== model I/O ==============================
def _load(model_id: str, attn_impl: str, need_attn: bool):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # FIX 1: Use 'dtype' instead of deprecated 'torch_dtype'
    kwargs = dict(
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    chosen_impl = None
    if attn_impl != "auto":
        chosen_impl = attn_impl
    elif need_attn:
        chosen_impl = "eager"

    # FIX 2: Translate shorthand 'fa2' to the full name required by the library
    if chosen_impl == "fa2":
        chosen_impl = "flash_attention_2"

    if chosen_impl is not None:
        try:
            kwargs["attn_implementation"] = chosen_impl
        except TypeError:
            pass

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    try:
        if need_attn and getattr(model.config, "attn_implementation", None) not in ("eager",):
            model.config.attn_implementation = "eager"
            warnings.warn("Forcing model.config.attn_implementation='eager' for attention outputs.")
    except Exception:
        pass
    _ensure_pad(model.config)
    model.eval()
    return model, tok

# ============================== forward paths ==============================
@torch.no_grad()
def prefill_forward(model, input_ids: torch.Tensor, cache: Cache, output_attentions: bool=False):
    B, S = input_ids.shape
    pos = torch.arange(S, device=input_ids.device, dtype=torch.long).unsqueeze(0)
    mask = torch.ones((B, S), device=input_ids.device, dtype=torch.long)
    out = model(input_ids=input_ids, position_ids=pos, attention_mask=mask,
                use_cache=True, past_key_values=cache, output_attentions=output_attentions)
    return S, out.attentions if output_attentions else None

@torch.no_grad()
def decode_from_cache(model, cache: Cache, last_token: torch.Tensor, start_abs_pos: int, n_new: int,
                      sample=False, temperature=0.7, top_p=0.9, top_k=0, need_attn=False):
    nxt = last_token
    out_tokens = []
    attn_last = None
    for t in range(n_new):
        pos = torch.tensor([[start_abs_pos + t]], device=nxt.device, dtype=torch.long)
        o = model(input_ids=nxt, position_ids=pos, use_cache=True, past_key_values=cache,
                  output_attentions=need_attn)
        logits = o.logits[:, -1, :]
        if need_attn:
            attn_last = o.attentions  # may be None if backend doesn't support
        if sample:
            if temperature and temperature > 0:
                logits = logits / max(1e-6, float(temperature))
            if top_k and 0 < top_k < logits.shape[-1]:
                vals, idx = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, idx, vals)
                logits = mask
            probs = torch.softmax(logits, dim=-1)
            if top_p and top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                csum = torch.cumsum(sp, dim=-1)
                cutoff = csum > float(top_p)
                cutoff[..., 0] = False
                sp = torch.where(cutoff, torch.zeros_like(sp), sp)
                sp = sp / sp.sum(dim=-1, keepdim=True)
                j = torch.multinomial(sp, 1)
                nxt = si.gather(1, j)
            else:
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        else:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        out_tokens.append(nxt)
        if need_attn and isinstance(cache, AttnPruneCache):
            cache.post_decode_prune(attn_last)
    return torch.cat(out_tokens, dim=1) if out_tokens else torch.empty_like(last_token)

# ============================== reporting ==============================
def print_tables(stats_prefill, stats_total):
    print("\nPer-layer retention — PREFILL ONLY:")
    for i,(s,k,r) in enumerate(stats_prefill):
        print(f"Layer {i:02d}: seen={s:5d}, kept={k:5d}, keep_rate={100*r:6.2f}%")
    print("\nPer-layer retention — TOTAL (prefill + decode):")
    for i,(s,k,r) in enumerate(stats_total):
        print(f"Layer {i:02d}: seen={s:5d}, kept={k:5d}, keep_rate={100*r:6.2f}%")

def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")

CSV_FIELDS = [
    "time", "cmd", "policy", "model",
    "ctx", "new", "block", "window",
    "anchors", "start_keep", "end_keep", "reservoir",
    "attn_impl",
    "throughput_toks_per_s", "peak_gpu_gb",
    "kv_current_mb_est", "kv_peak_mb_est",
    "kv_current_mb_exact",  # end-current exact (single scan post-decode)
    "ppl",
]

def write_csv_row(path: str, row: Dict[str, Any]):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})

# ============================== datasets for PPL ==============================
def _load_ppl_dataset(name: str) -> Tuple[str, str]:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("pip install datasets") from e

    name = (name or "").lower()
    if name in ("wikitext2","wikitext-2","wikitext-2-raw","wt2"):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        split = "validation" if "validation" in ds else "test"
        text = "\n".join(ds[split]["text"])
        return text, "dataset:wikitext-2-raw-v1"
    elif name in ("wikitext103","wikitext-103","wikitext-103-raw","wt103"):
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        split = "validation" if "validation" in ds else "test"
        text = "\n".join(ds[split]["text"])
        return text, "dataset:wikitext-103-raw-v1"
    elif name in ("ptb","penn","penn-treebank"):
        ds = load_dataset("ptb_text_only")
        split = "validation" if "validation" in ds else "test"
        text = "\n".join(ds[split]["sentence"])
        return text, "dataset:ptb_text_only"
    else:
        raise ValueError("Supported ppl datasets: wikitext2, wikitext103, ptb")

def _get_ppl_text(a) -> Tuple[str, str]:
    if a.ppl_dataset:
        return _load_ppl_dataset(a.ppl_dataset)
    if a.ppl_file and os.path.isfile(a.ppl_file):
        return open(a.ppl_file, "r", encoding="utf-8").read(), f"file:{a.ppl_file}"
    if a.ppl_text:
        return a.ppl_text, "inline_text"
    return " ".join(["Bangkok is the capital of Thailand."] * 10000), "fallback_inline"

# ============================== CLI cmds ==============================
def _mk_cache(model, policy: str, args: PolicyArgs):
    L = getattr(model.config, "num_hidden_layers", None)
    H = getattr(model.config, "num_attention_heads", None)
    assert L and H, "Model config missing num_hidden_layers / num_attention_heads"
    if policy == "full":
        return FullCache(L,H,args), linear_schedule(args.start_keep, args.end_keep, L)
    elif policy in ("window","fifo"):
        return WindowCache(L,H,args), linear_schedule(1.0,1.0,L)
    elif policy == "prune":
        return PruneNormCache(L,H,args), linear_schedule(args.start_keep, args.end_keep, L)
    elif policy == "attn":
        return AttnPruneCache(L,H,args), linear_schedule(args.start_keep, args.end_keep, L)
    elif policy == "tskip":
        return TokenSkippingCache(L,H,args), linear_schedule(args.start_keep, args.end_keep, L)
    else:
        raise ValueError(f"Unknown policy: {policy}")

@torch.no_grad()
def cmd_bench(a):
    print(f"[RUN] bench | method={policy_label(a.policy)} ({a.policy}) | model={a.model} "
          f"| ctx={a.ctx} new={a.new} | anchors={a.anchors} start={a.start_keep} end={a.end_keep} "
          f"| window={a.window} | attn_impl={a.attn_impl}")

    need_attn = (a.policy == "attn")
    model, tok = _load(a.model, a.attn_impl, need_attn=need_attn)
    pargs = PolicyArgs(anchors=a.anchors, reservoir=a.reservoir, start_keep=a.start_keep,
                       end_keep=a.end_keep, debug_keep=False, stream_exact=a.stream_exact,
                       window=a.window)
    cache, schedule = _mk_cache(model, a.policy, pargs)

    # build context text
    base = "Bangkok is the capital of Thailand. "
    ctx_text = (base * max(1, a.ctx // len(base)))[:a.ctx]
    enc = tok(ctx_text, return_tensors="pt").to(model.device)

    # PREFILL (not timed)
    next_pos, _ = prefill_forward(model, enc["input_ids"], cache, output_attentions=False)

    # reset CUDA peak BEFORE timing decode
    _device_synchronize()
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    # DECODE timing (no memory scans inside)
    t0 = time.time()
    new_ids = decode_from_cache(model, cache, enc["input_ids"][:, -1:], next_pos, a.new,
                                sample=False, need_attn=need_attn)
    _device_synchronize()
    dt = time.time() - t0
    tps = new_ids.shape[1] / max(1e-6, dt)
    peak_gpu = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0

    # KV usage: fast, speed-neutral estimates (based on counters only)
    bptpl = _kv_bytes_per_token_per_layer(model)
    kv_cur_est, kv_peak_est = (0,0)
    if isinstance(cache, BasePolicyCache):
        kv_cur_est, kv_peak_est = cache.kv_estimated_bytes(bptpl)

    # Optional exact end-current scan (single pass AFTER timing)
    kv_cur_exact = ""
    if a.mem_measure in ("exact", "both"):
        kv_cur_exact = round(cache.kv_exact_end_bytes() / 1024**2, 3) if isinstance(cache, BasePolicyCache) else ""

    print(f"Throughput: {tps:.2f} toks/s | Peak GPU: {peak_gpu:.2f} GB")
    print(f"KV usage (fast-est): end-current={kv_cur_est/1024**2:.2f} MB | decode-peak={kv_peak_est/1024**2:.2f} MB")
    if kv_cur_exact != "":
        print(f"KV usage (exact): end-current={kv_cur_exact:.2f} MB")

    # CSV
    if a.csv:
        write_csv_row(a.csv, dict(
            time=now_iso(), cmd="bench", policy=a.policy, model=a.model,
            ctx=a.ctx, new=a.new, block="", window=a.window,
            anchors=a.anchors, start_keep=a.start_keep, end_keep=a.end_keep, reservoir=a.reservoir,
            attn_impl=a.attn_impl,
            throughput_toks_per_s=round(tps,3), peak_gpu_gb=round(peak_gpu,3),
            kv_current_mb_est=round(kv_cur_est/1024**2,3), kv_peak_mb_est=round(kv_peak_est/1024**2,3),
            kv_current_mb_exact=kv_cur_exact, ppl="",
        ))

@torch.no_grad()
def _chunked(seq: Iterable[int], n: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) == n:
            yield buf; buf=[]
    if buf: yield buf

@torch.no_grad()
def streaming_ppl(model, tok, text: str, block: int, cache_factory):
    """Calculates PPL by processing the text in large chunks. Suitable for most policies."""
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)[0]
    losses = []
    
    # FIX 1: Create the cache only once, before the loop, to make it persistent.
    cache = cache_factory()
    
    # FIX 2: Limit the number of chunks to prevent running out of VRAM.
    max_chunks = 30
    for i, chunk in enumerate(_chunked(ids.tolist(), block)):
        if i >= max_chunks:
            print(f"\n[INFO] Stopping PPL calculation after {max_chunks} chunks to conserve memory.")
            break
        
        ip = torch.tensor([chunk], device=model.device)
        out = model(input_ids=ip, labels=ip, use_cache=True, past_key_values=cache)
        losses.append(float(out.loss))
        
    mean_loss = sum(losses) / max(1, len(losses))
    return math.exp(mean_loss)

@torch.no_grad()
def autoregressive_ppl(model, tok, text: str, block: int, cache_factory):
    """A slower, token-by-token PPL calculation needed for post-decode policies like AttnPrune."""
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)[0]
    
    cache = cache_factory()
    all_losses = []
    
    max_chunks = 20 # A smaller limit as this method is slower and more memory intensive
    for i, chunk in enumerate(_chunked(ids.tolist(), block)):
        if i >= max_chunks:
            print(f"\n[INFO] Stopping PPL calculation after {max_chunks} chunks to conserve memory.")
            break
        
        chunk_ids = torch.tensor(chunk, device=model.device)
        
        # Process each chunk token-by-token
        for t in range(len(chunk_ids) - 1):
            current_token = chunk_ids[t].unsqueeze(0).unsqueeze(0) # Shape: [1, 1]
            next_token_label = chunk_ids[t+1].unsqueeze(0)        # Shape: [1]
            
            # FIX 3: Add `output_attentions=True` to ensure the policy can activate.
            out = model(input_ids=current_token, use_cache=True, past_key_values=cache, output_attentions=True)
            
            # Prune the cache after the forward pass
            if isinstance(cache, AttnPruneCache):
                cache.post_decode_prune(out.attentions)
            
            # Calculate loss for the next token prediction
            logits = out.logits[:, -1, :]
            loss = torch.nn.functional.cross_entropy(logits, next_token_label)
            all_losses.append(loss.item())

    mean_loss = sum(all_losses) / max(1, len(all_losses))
    return math.exp(mean_loss)

def cmd_ppl(a):
    text, src = _get_ppl_text(a)
    print(f"[RUN] ppl   | method={policy_label(a.policy)} ({a.policy}) | model={a.model} "
          f"| block={a.block} | source={src} | anchors={a.anchors} start={a.start_keep} end={a.end_keep} "
          f"| window={a.window} | attn_impl={a.attn_impl}")
    need_attn = (a.policy == "attn")
    model, tok = _load(a.model, a.attn_impl, need_attn=need_attn)
    pargs = PolicyArgs(anchors=a.anchors, reservoir=a.reservoir, start_keep=a.start_keep,
                       end_keep=a.end_keep, debug_keep=False, stream_exact=a.stream_exact,
                       window=a.window)
    factory = lambda: _mk_cache(model, a.policy, pargs)[0]
    
    # FIX 4: Use the correct PPL function based on the policy
    if a.policy == "attn":
        print("[INFO] Using token-by-token PPL calculation for Attention-score pruning.")
        ppl = autoregressive_ppl(model, tok, text, block=a.block, cache_factory=factory)
    else:
        ppl = streaming_ppl(model, tok, text, block=a.block, cache_factory=factory)

    print(f"Streaming PPL: {ppl:.3f}")
    if a.csv:
        write_csv_row(a.csv, dict(
            time=now_iso(), cmd="ppl", policy=a.policy, model=a.model,
            ctx="", new="", block=a.block, window=a.window,
            anchors=a.anchors, start_keep=a.start_keep, end_keep=a.end_keep, reservoir=a.reservoir,
            attn_impl=a.attn_impl,
            throughput_toks_per_s="", peak_gpu_gb="",
            kv_current_mb_est="", kv_peak_mb_est="", kv_current_mb_exact="",
            ppl=round(ppl,3),
        ))

def cmd_demo(a):
    print(f"[RUN] demo | method={policy_label(a.policy)} ({a.policy}) | model={a.model} "
          f"| new={a.new} | anchors={a.anchors} start={a.start_keep} end={a.end_keep} "
          f"| window={a.window} | attn_impl={a.attn_impl}")
    need_attn = (a.policy == "attn")
    model, tok = _load(a.model, a.attn_impl, need_attn=need_attn)
    pargs = PolicyArgs(anchors=a.anchors, reservoir=a.reservoir, start_keep=a.start_keep,
                       end_keep=a.end_keep, debug_keep=a.debug_keep,
                       stream_exact=a.stream_exact, window=a.window)
    cache, schedule = _mk_cache(model, a.policy, pargs)

    prompt = a.prompt or (
        "Hello from Bangkok! I'm a 20 year old student at the University of Oxford. "
        "I've been to Thailand twice, and I'm planning on going again this summer.\n"
        "I'm really interested in Thai culture and history, so I was wondering "
        "if anyone knows any good places to visit or things to do in Bangkok? Also, are"
    )
    enc = tok(prompt, return_tensors="pt").to(model.device)

    next_pos, _ = prefill_forward(model, enc["input_ids"], cache, output_attentions=False)

    out_ids = enc["input_ids"]
    if a.new > 0:
        new_only = decode_from_cache(model, cache, enc["input_ids"][:, -1:], next_pos, a.new,
                                     sample=a.sample, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                                     need_attn=need_attn)
        out_ids = torch.cat([out_ids, new_only], dim=1)

    print(tok.decode(out_ids[0], skip_special_tokens=True))
    stats_total = cache.retention_stats()
    print("\nPer-layer retention (total):")
    for i,(s,k,r) in enumerate(stats_total):
        print(f"Layer {i:02d}: seen={s:5d}, kept={k:5d}, keep_rate={100*r:6.2f}%")

# ============================== main ==============================
def main():
    ap = argparse.ArgumentParser("Speed-neutral KV benchmarks (TokenSkipping & baselines)")
    ap.add_argument("cmd", choices=["demo","bench","ppl"])
    ap.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--policy", type=str, default="tskip",
                    choices=["full","window","fifo","prune","attn","tskip"])
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--reservoir", type=int, default=64)
    ap.add_argument("--start-keep", type=float, default=0.75)
    ap.add_argument("--end-keep", type=float, default=0.50)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--attn-impl", type=str, default="eager",
                    choices=["auto","eager","sdpa","fa2"])
    ap.add_argument("--stream-exact", action="store_true")
    ap.add_argument("--debug-keep", action="store_true")

    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)

    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--block", type=int, default=512)

    ap.add_argument("--ppl-file", type=str, default="")
    ap.add_argument("--ppl-text", type=str, default="")
    ap.add_argument("--ppl-dataset", type=str, default="", choices=["","wikitext2","wikitext103","ptb"])

    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--mem-measure", type=str, default="fast", choices=["fast","exact","both"],
                    help="KV memory measurement: fast=counter-based (no scans), exact=scan end-current only, both=print both.")

    a = ap.parse_args()
    if a.policy == "fifo":
        a.policy = "window"

    if a.cmd == "bench": cmd_bench(a)
    elif a.cmd == "ppl": cmd_ppl(a)
    elif a.cmd == "demo": cmd_demo(a)

if __name__ == "__main__":
    main()