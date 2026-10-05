"""Replace the most predictable words with fitting but less probable alternatives.

For each word that the LM ranked as its top-1 prediction with high confidence, we ask the same LM
for alternatives at that position, complete them into whole words, and keep a replacement only if:
  * the rest of the sentence stays about as likely as before (grammar / agreement still fit),
  * the candidate itself is still reasonably plausible in context,
  * meaning is preserved (sentence- and word-level embedding similarity, when an embedder is loaded).
"""
import math
import random
from dataclasses import dataclass

from .textutil import STOPWORDS, match_case, script_of, split_sentences, words


@dataclass
class LexicalConfig:
    strength: float = 0.12        # max fraction of content words to replace
    min_prob: float = 0.25        # only touch words the LM predicted with p >= this (as top-1)
    topk: int = 30                # alternatives considered per position
    max_extra_tokens: int = 3     # greedy completion length for multi-token words
    ctx_tokens: int = 128         # left context fed to the LM for candidate generation
    right_tokens: int = 16        # right context used for the fluency check
    max_right_drop: float = 0.35  # allowed drop in mean log-prob of the right context (nats/token)
    min_cand_prob: float = 0.002  # candidate word must keep at least this probability
    min_sent_sim: float = 0.92    # sentence embedding similarity after swap
    min_word_sim: float = 0.45    # word-level embedding similarity (synonym-ness)
    min_gap_words: int = 3        # don't edit words closer than this to each other
    seed: int = 0


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.lower(), b.lower()):
        if x != y:
            break
        n += 1
    return n


def _is_word_piece(s: str) -> bool:
    s2 = s.replace("-", "")
    return bool(s2) and s2.isalpha()


def predictable_words(lm, text, stats, cfg):
    """Return candidate word slots sorted by how predictable they were."""
    offs, lp, rank = stats["offsets"], stats["logp"], stats["rank"]
    sents = split_sentences(text)
    sent_starts = {s for s, _ in sents}
    out = []
    ti = 0
    for wi, (ws, we, w) in enumerate(words(text)):
        while ti < len(offs) and offs[ti][1] <= ws:
            ti += 1
        toks = []
        j = ti
        while j < len(offs) and offs[j][0] < we:
            toks.append(j)
            j += 1
        if not toks:
            continue
        # token boundaries must line up with the word (tokens may include the leading space)
        if offs[toks[-1]][1] != we or text[offs[toks[0]][0]:ws].strip():
            continue
        if len(w) < 4 or w.lower() in STOPWORDS:
            continue
        if w[:1].isupper() and ws not in sent_starts:  # likely a proper noun / acronym
            continue
        t0 = toks[0]
        if math.isnan(lp[t0]) or rank[t0] != 0 or math.exp(lp[t0]) < cfg.min_prob:
            continue
        out.append({"wi": wi, "start": ws, "end": we, "word": w, "toks": toks, "p": math.exp(lp[t0])})
    out.sort(key=lambda d: -d["p"])
    return out


def _alternatives(lm, text, stats, slot, cfg, embedder):
    ids, offs = stats["ids"], stats["offsets"]
    t0, tl = slot["toks"][0], slot["toks"][-1]
    ctx = (lm.prefix + ids[:t0])[-cfg.ctx_tokens:]
    if not ctx:
        return []
    orig_word_ids = ids[t0:tl + 1]
    first_txt = lm.decode([ids[t0]])
    lead_space = first_txt[:1].isspace()

    # right context: tokens after the word up to sentence end / right_tokens
    right = []
    for j in range(tl + 1, min(len(ids), tl + 1 + cfg.right_tokens)):
        right.append(ids[j])
        if any(c in lm.decode([ids[j]]) for c in ".!?\n"):
            break

    cand_ids, cand_lp = lm.topk_next(ctx, cfg.topk)
    seeds = []
    for tid, l in zip(cand_ids, cand_lp):
        if tid == ids[t0]:
            continue
        s = lm.decode([tid])
        if lead_space != s[:1].isspace():
            continue
        if not _is_word_piece(s.strip()):
            continue
        seeds.append((tid, l))
    if not seeds:
        return []

    # complete multi-token words greedily
    ext = lm.greedy_extend([ctx + [t] for t, _ in seeds], cfg.max_extra_tokens) if cfg.max_extra_tokens else [[] for _ in seeds]
    cands = []
    for (tid, l), more in zip(seeds, ext):
        wid = [tid]
        for t in more:
            piece = lm.decode([t])
            if piece[:1].isspace() or not _is_word_piece(piece):
                break
            wid.append(t)
        else:
            continue  # never reached a word boundary -> unreliable
        cw = lm.decode(wid).strip()
        if len(cw) < 3 or script_of(cw) != script_of(slot["word"]):
            continue
        if cw.lower() == slot["word"].lower():
            continue
        if _common_prefix(cw, slot["word"]) >= min(5, len(slot["word"]) - 1):
            continue  # just another inflection of the same word
        if cw.lower() in text.lower()[max(0, slot["start"] - 200):slot["end"] + 200].split():
            continue  # avoid creating repetition
        cands.append((wid, cw))
    if not cands:
        return []

    # fluency: candidate prob + right-context prob vs. original
    scores = lm.continuation_logp([ctx] * (len(cands) + 1), [orig_word_ids + right] + [w + right for w, _ in cands])
    o_right = scores[0][len(orig_word_ids):]
    o_mean = sum(o_right) / max(1, len(o_right))
    good = []
    for (wid, cw), sc in zip(cands, scores[1:]):
        w_lp = sum(sc[:len(wid)])
        r = sc[len(wid):]
        r_mean = sum(r) / max(1, len(r))
        if w_lp < math.log(cfg.min_cand_prob) or r_mean < o_mean - cfg.max_right_drop:
            continue
        good.append({"word": cw, "word_lp": w_lp, "right_drop": o_mean - r_mean})
    if not good:
        return []

    if embedder is not None:
        sent = next(((a, b) for a, b in split_sentences(text) if a <= slot["start"] < b), (0, len(text)))
        orig_sent = text[sent[0]:sent[1]]
        new_sents = [
            text[sent[0]:slot["start"]] + match_case(slot["word"], g["word"]) + text[slot["end"]:sent[1]] for g in good
        ]
        ssims = embedder.sims_to(orig_sent, new_sents)
        wsims = embedder.sims_to(slot["word"].lower(), [g["word"].lower() for g in good])
        for g, a, b in zip(good, ssims, wsims):
            g["sent_sim"], g["word_sim"] = a, b
        good = [g for g in good if g["sent_sim"] >= cfg.min_sent_sim and g["word_sim"] >= cfg.min_word_sim]
        good.sort(key=lambda g: (-(g["word_sim"] + g["sent_sim"]), g["word_lp"]))
    else:
        # LM-only: prefer the candidate that disturbs the rest of the sentence least
        good.sort(key=lambda g: (g["right_drop"], -g["word_lp"]))
    return good


def lexical_pass(lm, text: str, cfg: LexicalConfig | None = None, embedder=None, log=None):
    cfg = cfg or LexicalConfig()
    rnd = random.Random(cfg.seed)
    stats = lm.token_stats(text)
    slots = predictable_words(lm, text, stats, cfg)
    n_content = sum(1 for _, _, w in words(text) if w.lower() not in STOPWORDS)
    budget = max(1, int(cfg.strength * n_content))

    edits, used = [], set()
    for slot in slots:
        if len(edits) >= budget:
            break
        if any(abs(slot["wi"] - u) < cfg.min_gap_words for u in used):
            continue
        alts = _alternatives(lm, text, stats, slot, cfg, embedder)
        if not alts:
            continue
        # small randomness among the top choices so repeated runs don't produce a fixed pattern
        pick = alts[0] if len(alts) == 1 or rnd.random() < 0.7 else alts[1]
        edits.append((slot["start"], slot["end"], match_case(slot["word"], pick["word"])))
        used.add(slot["wi"])
        if log is not None:
            log.append({"from": slot["word"], "to": pick["word"], "p_orig": round(slot["p"], 3),
                        **{k: round(v, 3) for k, v in pick.items() if isinstance(v, float)}})

    for s, e, w in sorted(edits, reverse=True):
        text = text[:s] + w + text[e:]
    return text
