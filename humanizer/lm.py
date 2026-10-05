"""Thin wrapper around a causal language model used both to measure and to steer rewriting."""
import math
import os

import torch
import torch.nn.functional as F

DEFAULT_LM = os.environ.get("HUMANIZER_LM", "Qwen/Qwen2.5-0.5B")


class LM:
    def __init__(self, name: str = DEFAULT_LM, device: str | None = None, window: int = 1024, overlap: int = 128):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(self.device).eval()
        self.window, self.overlap = window, overlap
        bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        self.prefix = [bos] if bos is not None else []
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.tok.eos_token_id or 0)

    # ---------- tokenization ----------
    def encode(self, text: str):
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

    def decode(self, ids) -> str:
        return self.tok.decode(ids, clean_up_tokenization_spaces=False)

    # ---------- forward helpers ----------
    @torch.no_grad()
    def _logits(self, batch_ids, attn=None):
        x = torch.tensor(batch_ids, device=self.device)
        a = torch.tensor(attn, device=self.device) if attn is not None else None
        return self.model(input_ids=x, attention_mask=a).logits.float()

    @torch.no_grad()
    def token_stats(self, text: str):
        """Per-token log-prob, rank, and the analytic mean/variance of log-prob under the model.

        Returns dict with lists aligned to the text's tokens (first token may be NaN if the
        tokenizer has no BOS/EOS to condition on)."""
        ids, offs = self.encode(text)
        p = len(self.prefix)
        full = self.prefix + ids
        n = len(ids)
        lp = [math.nan] * n
        rank = [-1] * n
        mu = [math.nan] * n
        var = [math.nan] * n
        ent = [math.nan] * n
        first = 0 if p else 1
        chunk = max(16, self.window - self.overlap - 1)
        a = first
        while a < n:
            b = min(n, a + chunk)
            start = max(0, a + p - 1 - self.overlap)
            logits = self._logits([full[start:b + p]])[0]
            rows = torch.arange(a + p - 1 - start, b + p - 1 - start, device=logits.device)
            logp = F.log_softmax(logits[rows], dim=-1)
            tgt = torch.tensor(ids[a:b], device=logits.device)
            t_lp = logp.gather(1, tgt[:, None])[:, 0]
            probs = logp.exp()
            m = (probs * logp).sum(-1)
            v = (probs * logp.pow(2)).sum(-1) - m.pow(2)
            r = (logp > t_lp[:, None]).sum(-1)
            for k, i in enumerate(range(a, b)):
                lp[i] = t_lp[k].item()
                rank[i] = int(r[k].item())
                mu[i] = m[k].item()
                var[i] = v[k].item()
                ent[i] = -m[k].item()
            a = b
        return {"ids": ids, "offsets": offs, "logp": lp, "rank": rank, "mu": mu, "var": var, "entropy": ent}

    @torch.no_grad()
    def topk_next(self, ctx_ids, k: int):
        logits = self._logits([ctx_ids])[0, -1]
        logp = F.log_softmax(logits, -1)
        vals, idx = logp.topk(k)
        return idx.tolist(), vals.tolist()

    @torch.no_grad()
    def greedy_extend(self, seqs, steps: int):
        """seqs: equal-length id lists. Returns list of lists of greedily generated ids (len==steps)."""
        cur = [list(s) for s in seqs]
        gen = [[] for _ in seqs]
        for _ in range(steps):
            logits = self._logits(cur)[:, -1]
            nxt = logits.argmax(-1).tolist()
            for i, t in enumerate(nxt):
                cur[i].append(t)
                gen[i].append(t)
        return gen

    @torch.no_grad()
    def continuation_logp(self, ctxs, conts):
        """Batched: for each (ctx, cont) returns list of per-token log-probs of cont given ctx."""
        seqs = [c + d for c, d in zip(ctxs, conts)]
        L = max(len(s) for s in seqs)
        batch = [s + [self.pad_id] * (L - len(s)) for s in seqs]
        attn = [[1] * len(s) + [0] * (L - len(s)) for s in seqs]
        logp = F.log_softmax(self._logits(batch, attn), -1)
        out = []
        for i, (c, d) in enumerate(zip(ctxs, conts)):
            pos = torch.arange(len(c) - 1, len(c) + len(d) - 1, device=logp.device)
            tg = torch.tensor(d, device=logp.device)
            out.append(logp[i, pos].gather(1, tg[:, None])[:, 0].tolist())
        return out
