"""Profile matching: pick, sentence by sentence, the variant that brings the whole text's
statistical profile closest to a target human profile, under a meaning-preservation constraint."""
import json
import random
import re

from . import llm_rewrite
from .lexical import LexicalConfig, lexical_pass
from .profile import build_profile, distance
from .structure import remove_cliches
from .textutil import split_sentences


class Doc:
    """A text as sentences plus the exact separators between them, so variants can be swapped in."""

    def __init__(self, text):
        spans = split_sentences(text)
        self.sents = [text[a:b] for a, b in spans]
        self.seps = [text[:spans[0][0]] if spans else text]
        for (a, b), (c, _) in zip(spans, spans[1:]):
            self.seps.append(text[b:c])
        self.seps.append(text[spans[-1][1]:] if spans else "")

    def render(self, sents):
        out = [self.seps[0]]
        for i, s in enumerate(sents):
            out.append(s)
            out.append(self.seps[i + 1])
        return "".join(out)


# ---------------- candidate generators ----------------

VARIANT_PROMPT = """Below is a text split into numbered sentences. For EACH sentence write {k} alternative versions \
that a human writer might plausibly have written instead, in {lang_name}. Keep the meaning, facts, numbers and names; \
fit the surrounding context. Spread the versions across different kinds of change:
1. a light edit (a word or two swapped for less obvious but natural ones),
2. a restructured sentence (different word order or syntax),
3. a different length (split into two short sentences, or fold in a clause to make it longer and more winding),
4. a plainer, more conversational phrasing for the register.
Avoid stock AI connectors ({cliches}).

Reply with only a JSON object mapping each sentence number (as a string) to a list of {k} strings.

Sentences:
{items}
"""


def claude_variants(sents, lang, k=4):
    import anthropic

    items = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))
    prompt = VARIANT_PROMPT.format(k=k, lang_name=llm_rewrite.LANG_NAME.get(lang, lang),
                                   cliches=llm_rewrite.CLICHE_HINT.get(lang, ""), items=items)
    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=llm_rewrite.MODEL,
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},
        max_tokens=32000,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "medium"},
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("Claude declined to produce variants")
    raw = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group(0)) if m else {}
    return [[v for v in data.get(str(i + 1), []) if isinstance(v, str) and v.strip()] for i in range(len(sents))]


def local_variants(lm, text, lang, k=4, embedder=None, seed=0):
    """Variants without an API: LM-guided word swaps at several strengths + cliché rewrites per sentence."""
    doc = Doc(text)
    out = [[] for _ in doc.sents]
    # without an embedder, LM-only word swaps are unchecked for meaning: keep only cliché rewrites
    for j in range(k if embedder is not None else 0):
        cfg = LexicalConfig(strength=0.08 + 0.06 * j, seed=seed + j, min_gap_words=2)
        vt = lexical_pass(lm, text, cfg, embedder)
        vd = Doc(vt)
        if len(vd.sents) == len(doc.sents):
            for i, s in enumerate(vd.sents):
                out[i].append(s)
    rnd = random.Random(seed)
    for i, s in enumerate(doc.sents):
        out[i].append(remove_cliches(s, lang, rnd))
    return out


PARA_PROMPTS = {
    "ru": ["Перепиши предложение другими словами, сохранив смысл и все факты. Сделай его проще и короче. Ответь только новым предложением.",
           "Перепиши предложение, полностью изменив порядок слов и построение фразы, но сохранив смысл и все факты. Ответь только новым предложением.",
           "Перескажи это предложение живым разговорным языком, как написал бы человек, сохранив смысл. Ответь только новым текстом."],
    "en": ["Rewrite the sentence in different words, keeping its meaning and all facts. Make it simpler and shorter. Reply with the new sentence only.",
           "Rewrite the sentence with a completely different word order and structure, keeping its meaning and all facts. Reply with the new sentence only.",
           "Retell this sentence in plain, natural language the way a person would write it, keeping the meaning. Reply with the new text only."],
}


def local_llm_variants(text, lang, model_name="Qwen/Qwen2.5-0.5B-Instruct", k=3, seed=0):
    """Fully offline variants from a small local instruct model (same one the offline web app uses)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    sents = Doc(text).sents
    out = []
    for i, s in enumerate(sents):
        vs = []
        for j in range(k):
            ctx = (("Контекст: " if lang == "ru" else "Context: ") + sents[i - 1] + "\n") if i else ""
            msgs = [{"role": "system", "content": PARA_PROMPTS[lang][j % 3]},
                    {"role": "user", "content": ctx + ("Предложение: " if lang == "ru" else "Sentence: ") + s}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=min(160, int(len(tok(s)["input_ids"]) * 1.8) + 8),
                                   do_sample=True, temperature=0.8, top_p=0.9)
            txt = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip(' "«»')
            r = len(txt.split()) / max(1, len(s.split()))
            if txt and txt != s and 0.4 < r < 2.2:
                vs.append(txt)
        out.append(vs)
    return out


# ---------------- search ----------------

def profile_match(lm, text, target, variants, embedder=None, min_sent_sim=0.85, sweeps=2, seed=0, log=None):
    """Coordinate descent over per-sentence choices minimizing distance(profile(text), target)."""
    doc = Doc(text)
    n = len(doc.sents)
    cands = []
    for i in range(n):
        opts = [doc.sents[i]]
        for v in variants[i] if i < len(variants) else []:
            v = v.strip()
            if v and v not in opts:
                opts.append(v)
        if embedder is not None and len(opts) > 1:
            sims = embedder.sims_to(doc.sents[i], opts[1:])
            opts = [opts[0]] + [o for o, s in zip(opts[1:], sims) if s >= min_sent_sim]
        cands.append(opts)

    choice = [0] * n
    cache = {}

    def score(ch):
        key = tuple(ch)
        if key not in cache:
            t = doc.render([cands[i][c] for i, c in enumerate(ch)])
            cache[key] = distance(build_profile(lm, t), target)
        return cache[key]

    best = score(choice)
    start = best
    rnd = random.Random(seed)
    for sweep in range(sweeps):
        improved = False
        order = list(range(n))
        rnd.shuffle(order)
        for i in order:
            for j in range(len(cands[i])):
                if j == choice[i]:
                    continue
                trial = choice[:]
                trial[i] = j
                d = score(trial)
                if d < best - 1e-6:
                    best, choice, improved = d, trial, True
        if log is not None:
            log.append({"sweep": sweep + 1, "distance": round(best, 3)})
        if not improved:
            break
    final = doc.render([cands[i][c] for i, c in enumerate(choice)])
    return final, {"distance_before": round(start, 3), "distance_after": round(best, 3),
                   "changed_sentences": sum(1 for c in choice if c), "sentences": n,
                   "candidates": [len(c) - 1 for c in cands], "evaluations": len(cache)}
