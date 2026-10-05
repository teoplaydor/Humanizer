"""Orchestrates the passes and measures before/after."""
import math
import statistics as st
from dataclasses import dataclass, field

from . import llm_rewrite
from .lexical import LexicalConfig, lexical_pass
from .metrics import measure
from .structure import structure_pass
from .textutil import detect_lang, split_sentences


@dataclass
class HumanizeConfig:
    lang: str = "auto"
    use_llm: bool = False          # full Claude rewrite first
    llm_focus_rounds: int = 0      # extra Claude rounds on the most predictable sentences
    focus_k: int = 3
    structure: bool = True
    lexical: bool = True
    lexical_without_embedder: bool = False  # unsafe: LM-only swaps can change meaning
    lexical_cfg: LexicalConfig = field(default_factory=LexicalConfig)
    min_text_sim: float = 0.85     # revert a pass if whole-text meaning similarity drops below this
    seed: int = 0


def most_predictable_sentences(lm, text, k):
    s = lm.token_stats(text)
    scored = []
    for a, b in split_sentences(text):
        toks = [i for i, o in enumerate(s["offsets"]) if o[1] > a and o[0] < b and not math.isnan(s["logp"][i])]
        if len(toks) >= 5:
            scored.append((st.mean(-s["logp"][i] for i in toks), text[a:b]))
    scored.sort()
    return [t for _, t in scored[:k]]


def humanize(text: str, lm, embedder=None, cfg: HumanizeConfig | None = None):
    cfg = cfg or HumanizeConfig()
    lang = detect_lang(text) if cfg.lang == "auto" else cfg.lang
    report = {"lang": lang, "lm": lm.name, "passes": [], "lexical_edits": []}
    report["before"] = measure(lm, text)
    orig = text

    def accept(name, new):
        nonlocal text
        if embedder is not None:
            sim = embedder.text_sim(orig, new)
            if sim < cfg.min_text_sim:
                report["passes"].append({"pass": name, "status": f"reverted (meaning sim {sim:.3f})"})
                return
            report["passes"].append({"pass": name, "status": "ok", "meaning_sim": round(sim, 3)})
        else:
            report["passes"].append({"pass": name, "status": "ok"})
        text = new

    if cfg.use_llm:
        if llm_rewrite.available():
            accept("llm_rewrite", llm_rewrite.rewrite(text, lang))
        else:
            report["passes"].append({"pass": "llm_rewrite", "status": "skipped (no ANTHROPIC_API_KEY)"})

    if cfg.structure:
        accept("structure", structure_pass(text, lang, seed=cfg.seed))

    if cfg.lexical and embedder is None and not cfg.lexical_without_embedder:
        report["passes"].append({"pass": "lexical", "status": "skipped (no embedder: word swaps would be unchecked for meaning)"})
    elif cfg.lexical:
        cfg.lexical_cfg.seed = cfg.seed
        accept("lexical", lexical_pass(lm, text, cfg.lexical_cfg, embedder, log=report["lexical_edits"]))

    for r in range(cfg.llm_focus_rounds):
        if not llm_rewrite.available():
            report["passes"].append({"pass": f"llm_focus_{r + 1}", "status": "skipped (no ANTHROPIC_API_KEY)"})
            break
        focus = most_predictable_sentences(lm, text, cfg.focus_k)
        accept(f"llm_focus_{r + 1}", llm_rewrite.rewrite(text, lang, focus_sentences=focus))

    report["after"] = measure(lm, text)
    if embedder is not None:
        report["meaning_similarity"] = round(embedder.text_sim(orig, text), 3)
    return text, report


def humanize_to_profile(text: str, lm, target, embedder=None, generator: str = "auto", k: int = 4,
                        sweeps: int = 3, lang: str = "auto", seed: int = 0, eval_lm=None):
    """Research-grounded mode: move the text's statistical profile toward a human target profile."""
    from .profile import build_profile, distance
    from .search import Doc, claude_variants, local_llm_variants, local_variants, profile_match

    lang = detect_lang(text) if lang == "auto" else lang
    gen = generator
    if gen == "auto":
        gen = "claude" if llm_rewrite.available() else "local-llm"
    report = {"mode": "profile", "lang": lang, "lm": lm.name, "generator": gen}
    p0 = build_profile(lm, text)
    report["target"] = target.summary()
    report["before_profile"] = p0.summary()
    report["before_distance_parts"] = {k_: round(v, 3) for k_, v in distance(p0, target, detail=True)[1].items()}

    if gen == "claude":
        variants = claude_variants(Doc(text).sents, lang, k)
    elif gen == "local-llm":
        variants = local_llm_variants(text, lang, k=k, seed=seed)
        extra = local_variants(lm, text, lang, k, embedder, seed)   # add word-swap and cliché variants
        variants = [a + b for a, b in zip(variants, extra)]
    else:
        variants = local_variants(lm, text, lang, k, embedder, seed)
    log = []
    out, info = profile_match(lm, text, target, variants, embedder, sweeps=sweeps, seed=seed, log=log)
    report.update(info)
    report["sweeps"] = log
    p1 = build_profile(lm, out)
    report["after_profile"] = p1.summary()
    report["after_distance_parts"] = {k_: round(v, 3) for k_, v in distance(p1, target, detail=True)[1].items()}
    report["before"] = measure(lm, text)
    report["after"] = measure(lm, out)
    if eval_lm is not None:  # held-out model: was the gain real or just fitted to the optimizing LM?
        report["heldout_lm"] = eval_lm.name
        report["heldout_before"] = measure(eval_lm, text)
        report["heldout_after"] = measure(eval_lm, out)
    if embedder is not None:
        report["meaning_similarity"] = round(embedder.text_sim(text, out), 3)
    return out, report
