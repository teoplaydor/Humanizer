"""Statistics that AI-text detectors rely on, measured with a local causal LM."""
import math
import statistics as st

from .textutil import split_sentences, words


def _cv(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2:
        return 0.0
    m = st.mean(xs)
    return st.pstdev(xs) / m if m else 0.0


def measure(lm, text: str, stats=None) -> dict:
    s = stats or lm.token_stats(text)
    lp, rank, mu, var, offs = s["logp"], s["rank"], s["mu"], s["var"], s["offsets"]
    valid = [i for i in range(len(lp)) if not math.isnan(lp[i])]
    if not valid:
        return {}
    surpr = [-lp[i] for i in valid]
    mean_nll = st.mean(surpr)

    # Fast-DetectGPT analytic criterion: how much more likely the text is than a typical sample
    # from the model itself. Higher = more "model-like".
    num = sum(lp[i] - mu[i] for i in valid)
    den = math.sqrt(max(1e-9, sum(var[i] for i in valid)))
    fdg = num / den

    # Per-sentence perplexity -> burstiness
    sent_ppl, sent_len = [], []
    for a, b in split_sentences(text):
        toks = [i for i in valid if offs[i][1] > a and offs[i][0] < b]
        if len(toks) >= 3:
            sent_ppl.append(math.exp(st.mean(-lp[i] for i in toks)))
        sent_len.append(len(words(text[a:b])))

    return {
        "tokens": len(valid),
        "perplexity": round(math.exp(mean_nll), 2),
        "top1_share": round(sum(1 for i in valid if rank[i] == 0) / len(valid), 3),
        "top10_share": round(sum(1 for i in valid if rank[i] < 10) / len(valid), 3),
        "mean_entropy": round(st.mean(s["entropy"][i] for i in valid), 3),
        "surprisal_std": round(st.pstdev(surpr), 3),
        "burstiness_ppl_cv": round(_cv(sent_ppl), 3),
        "sentences": len(sent_len),
        "sent_len_mean": round(st.mean(sent_len), 1) if sent_len else 0,
        "sent_len_cv": round(_cv(sent_len), 3),
        "fast_detectgpt": round(fdg, 3),
    }


# Direction in which a metric moves when text becomes more "human".
HUMAN_DIRECTION = {
    "perplexity": +1,
    "top1_share": -1,
    "top10_share": -1,
    "mean_entropy": 0,
    "surprisal_std": +1,
    "burstiness_ppl_cv": +1,
    "sent_len_cv": +1,
    "fast_detectgpt": -1,
}

LABELS = {
    "perplexity": "Perplexity (выше = менее предсказуемо)",
    "top1_share": "Доля токенов = top-1 предсказание",
    "top10_share": "Доля токенов в top-10",
    "mean_entropy": "Средняя энтропия модели",
    "surprisal_std": "Разброс «удивления» токенов",
    "burstiness_ppl_cv": "Burstiness: CV perplexity по предложениям",
    "sent_len_cv": "CV длины предложений",
    "sent_len_mean": "Средняя длина предложения, слов",
    "fast_detectgpt": "Fast-DetectGPT (ниже = человечнее)",
}


def compare_table(before: dict, after: dict) -> str:
    rows = ["| Метрика | До | После | |", "|---|---|---|---|"]
    for k, label in LABELS.items():
        if k not in before or k not in after:
            continue
        d = HUMAN_DIRECTION.get(k, 0)
        delta = after[k] - before[k]
        mark = "" if d == 0 or abs(delta) < 1e-9 else ("✅" if delta * d > 0 else "⚠️")
        rows.append(f"| {label} | {before[k]} | {after[k]} | {mark} |")
    return "\n".join(rows)
