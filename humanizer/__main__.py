"""CLI.

    python -m humanizer measure input.txt
    python -m humanizer humanize input.txt -o output.txt [--llm] [--strength 0.12] [--report report.json]
"""
import argparse
import json
import sys

from .lexical import LexicalConfig
from .lm import DEFAULT_LM, LM
from .metrics import LABELS, compare_table, measure
from .pipeline import HumanizeConfig, humanize
from .semantic import DEFAULT_EMBED, load_embedder


def _read(path):
    return sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="humanizer")
    ap.add_argument("--lm", default=DEFAULT_LM, help="HF id or local path of the scoring causal LM")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="print detector-style statistics of a text")
    m.add_argument("input")

    pb = sub.add_parser("profile", help="build a target profile from human-written samples")
    pb.add_argument("samples", nargs="+", help="text files; separate several texts in one file with two blank lines")
    pb.add_argument("--out", required=True)

    h = sub.add_parser("humanize", help="rewrite a text")
    h.add_argument("input")
    h.add_argument("-o", "--output", default="-")
    h.add_argument("--lang", default="auto", choices=["auto", "ru", "en"])
    h.add_argument("--llm", action="store_true", help="start with a full Claude rewrite (needs ANTHROPIC_API_KEY)")
    h.add_argument("--llm-focus-rounds", type=int, default=0, help="extra Claude rounds on the most predictable sentences")
    h.add_argument("--strength", type=float, default=0.12, help="max share of content words to swap")
    h.add_argument("--no-structure", action="store_true")
    h.add_argument("--no-lexical", action="store_true")
    h.add_argument("--embed", default=DEFAULT_EMBED, help="sentence-embedding model for meaning checks, or 'none'")
    h.add_argument("--unsafe-lexical", action="store_true", help="allow word swaps even without an embedder")
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--report", help="write JSON report here")
    h.add_argument("--style", nargs="+", help="samples of human (ideally your own) writing: enables profile-matching mode")
    h.add_argument("--target", help="target profile JSON from `profile` (alternative to --style)")
    h.add_argument("--generator", default="auto", choices=["auto", "claude", "local-llm", "local"], help="where sentence variants come from")
    h.add_argument("--variants", type=int, default=4)
    h.add_argument("--sweeps", type=int, default=3)
    h.add_argument("--eval-lm", help="second, held-out LM for honest before/after measurement")
    a = ap.parse_args(argv)

    lm = LM(a.lm)
    text = _read(a.input) if getattr(a, "input", None) else ""

    if a.cmd == "measure":
        res = measure(lm, text)
        for k, v in res.items():
            print(f"{LABELS.get(k, k):45s} {v}")
        return

    if a.cmd == "profile":
        from .profile import build_profile, load_texts
        p = build_profile(lm, load_texts(a.samples))
        p.save(a.out)
        for k, v in p.summary().items():
            print(f"{k:16s} {v}")
        return

    emb = load_embedder(a.embed)
    if a.style or a.target:
        from .pipeline import humanize_to_profile
        from .profile import Profile, build_profile, load_texts
        target = Profile.load(a.target) if a.target else build_profile(lm, load_texts(a.style))
        if target.n_tokens < 300:
            sys.exit(f"Образец стиля слишком короткий ({target.n_tokens} токенов). Нужно хотя бы 300, лучше несколько тысяч.")
        eval_lm = LM(a.eval_lm) if a.eval_lm else None
        out, rep = humanize_to_profile(text, lm, target, emb, a.generator, a.variants, a.sweeps, a.lang, a.seed, eval_lm)
        if a.output == "-":
            print(out)
        else:
            open(a.output, "w", encoding="utf-8").write(out)
        print(f"\nРасстояние до профиля: {rep['distance_before']} -> {rep['distance_after']} "
              f"(изменено предложений: {rep['changed_sentences']} из {rep['sentences']})", file=sys.stderr)
        print("\n" + compare_table(rep["before"], rep["after"]), file=sys.stderr)
        if "heldout_after" in rep:
            print(f"\nНезависимая модель {rep['heldout_lm']}:\n" + compare_table(rep["heldout_before"], rep["heldout_after"]), file=sys.stderr)
        if a.report:
            json.dump(rep, open(a.report, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return
    cfg = HumanizeConfig(
        lang=a.lang, use_llm=a.llm, llm_focus_rounds=a.llm_focus_rounds,
        structure=not a.no_structure, lexical=not a.no_lexical, lexical_without_embedder=a.unsafe_lexical,
        lexical_cfg=LexicalConfig(strength=a.strength), seed=a.seed,
    )
    out, rep = humanize(text, lm, emb, cfg)
    if a.output == "-":
        print(out)
    else:
        open(a.output, "w", encoding="utf-8").write(out)
    print("\n" + compare_table(rep["before"], rep["after"]), file=sys.stderr)
    if "meaning_similarity" in rep:
        print(f"\nСмысловое сходство с оригиналом: {rep['meaning_similarity']}", file=sys.stderr)
    for p in rep["passes"]:
        print(f"  {p}", file=sys.stderr)
    if a.report:
        json.dump(rep, open(a.report, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
