"""Statistical writing profile and the distance between profiles.

A profile captures what zero-shot detectors look at, as distributions rather than single numbers:
  * GLTR-style histogram of token ranks under the scoring LM,
  * the distribution of token surprisal (DetectGPT / Fast-DetectGPT territory),
  * per-token Fast-DetectGPT discrepancy (log p - E[log p]) / sqrt(Var), length-normalized,
  * how surprisal fluctuates between sentences: coefficient of variation and lag-1 autocorrelation
    (GPTZero burstiness, DivEye-style variability),
  * sentence-length distribution and punctuation habits.

The humanizer moves a text toward a *target* profile built from real human writing (ideally the
user's own), instead of just pushing a detector score down.
"""
import json
import math
import re
import statistics as st
from dataclasses import asdict, dataclass, field

from .textutil import split_sentences, words

RANK_BINS = [(0, 0), (1, 2), (3, 9), (10, 99), (100, 10**9)]
PUNCT = {"comma": ",", "dash": "—–", "colon": ":;", "question": "?!", "paren": "()"}


@dataclass
class Profile:
    n_tokens: int = 0
    rank_hist: list = field(default_factory=lambda: [0.0] * len(RANK_BINS))
    surprisal: list = field(default_factory=list)       # token-level, sampled
    fdg_per_token: float = 0.0
    sent_surprisal: list = field(default_factory=list)  # mean surprisal per sentence
    sent_len: list = field(default_factory=list)
    punct: dict = field(default_factory=dict)            # share of sentences containing each mark

    # ---- derived ----
    @property
    def burst_cv(self):
        return _cv(self.sent_surprisal)

    @property
    def burst_ac1(self):
        return _ac1(self.sent_surprisal)

    def summary(self):
        return {
            "tokens": self.n_tokens,
            "top1": round(self.rank_hist[0], 3),
            "top10": round(sum(self.rank_hist[:3]), 3),
            "mean_surprisal": round(st.mean(self.surprisal), 3) if self.surprisal else 0,
            "fdg_per_token": round(self.fdg_per_token, 3),
            "burst_cv": round(self.burst_cv, 3),
            "burst_ac1": round(self.burst_ac1, 3),
            "sent_len_mean": round(st.mean(self.sent_len), 1) if self.sent_len else 0,
            "sent_len_cv": round(_cv(self.sent_len), 3),
        }

    def save(self, path):
        json.dump(asdict(self), open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        return cls(**json.load(open(path, encoding="utf-8")))


def _cv(xs):
    if len(xs) < 2:
        return 0.0
    m = st.mean(xs)
    return st.pstdev(xs) / m if m else 0.0


def _ac1(xs):
    if len(xs) < 3:
        return 0.0
    m = st.mean(xs)
    den = sum((x - m) ** 2 for x in xs)
    return sum((xs[i] - m) * (xs[i + 1] - m) for i in range(len(xs) - 1)) / den if den else 0.0


def build_profile(lm, texts, stats_list=None) -> Profile:
    """Profile of one text or of a collection (pooled)."""
    if isinstance(texts, str):
        texts = [texts]
    p = Profile(punct={k: 0.0 for k in PUNCT})
    counts = [0] * len(RANK_BINS)
    fdg_num = fdg_den = 0.0
    n_sent = 0
    for ti, text in enumerate(texts):
        s = stats_list[ti] if stats_list else lm.token_stats(text)
        lp, rank, mu, var, offs = s["logp"], s["rank"], s["mu"], s["var"], s["offsets"]
        valid = [i for i in range(len(lp)) if not math.isnan(lp[i])]
        for i in valid:
            for b, (lo, hi) in enumerate(RANK_BINS):
                if lo <= rank[i] <= hi:
                    counts[b] += 1
                    break
            p.surprisal.append(-lp[i])
            fdg_num += lp[i] - mu[i]
            fdg_den += var[i]
        p.n_tokens += len(valid)
        for a, b in split_sentences(text):
            toks = [i for i in valid if offs[i][1] > a and offs[i][0] < b]
            sent = text[a:b]
            if len(toks) >= 3:
                p.sent_surprisal.append(st.mean(-lp[i] for i in toks))
            p.sent_len.append(len(words(sent)))
            n_sent += 1
            for k, chars in PUNCT.items():
                if any(c in sent for c in chars):
                    p.punct[k] += 1
    if p.n_tokens:
        p.rank_hist = [c / p.n_tokens for c in counts]
        # per-token discrepancy: the Fast-DetectGPT z-score grows like sqrt(n), so divide it out
        p.fdg_per_token = (fdg_num / math.sqrt(max(fdg_den, 1e-9))) / math.sqrt(p.n_tokens)
    if n_sent:
        p.punct = {k: v / n_sent for k, v in p.punct.items()}
    return p


def _w1(a, b):
    """1-Wasserstein distance between two empirical samples (quantile matching)."""
    if not a or not b:
        return 0.0
    a, b = sorted(a), sorted(b)
    q = 64
    qa = [a[min(len(a) - 1, int(i / q * len(a)))] for i in range(q)]
    qb = [b[min(len(b) - 1, int(i / q * len(b)))] for i in range(q)]
    return sum(abs(x - y) for x, y in zip(qa, qb)) / q


DEFAULT_WEIGHTS = {
    "rank_hist": 2.0,      # GLTR
    "surprisal": 1.0,      # level/shape of surprisal
    "fdg": 1.5,            # Fast-DetectGPT
    "burst_cv": 1.0,       # GPTZero burstiness
    "burst_ac1": 0.5,      # rhythm (DivEye-style)
    "sent_len": 0.7,
    "punct": 0.5,
}


def distance(p: Profile, t: Profile, weights=None, detail=False):
    w = weights or DEFAULT_WEIGHTS
    s_scale = (st.pstdev(t.surprisal) if len(t.surprisal) > 1 else 1.0) or 1.0
    l_scale = (st.mean(t.sent_len) if t.sent_len else 10.0) or 10.0
    parts = {
        "rank_hist": sum(abs(x - y) for x, y in zip(p.rank_hist, t.rank_hist)),
        "surprisal": _w1(p.surprisal, t.surprisal) / s_scale,
        "fdg": abs(p.fdg_per_token - t.fdg_per_token) / 0.1,
        "burst_cv": abs(p.burst_cv - t.burst_cv) / 0.1,
        "burst_ac1": abs(p.burst_ac1 - t.burst_ac1) / 0.3,
        "sent_len": _w1(p.sent_len, t.sent_len) / l_scale,
        "punct": sum(abs(p.punct.get(k, 0) - t.punct.get(k, 0)) for k in PUNCT),
    }
    total = sum(w[k] * v for k, v in parts.items())
    return (total, parts) if detail else total


def load_texts(paths):
    out = []
    for p in paths:
        txt = open(p, encoding="utf-8").read()
        out.extend(x for x in re.split(r"\n\s*\n\s*\n", txt) if x.strip())
    return out
