"""Rule-based structural variation: strip LLM clichés and vary sentence length (burstiness)."""
import random
import re
import statistics as st

from .textutil import STOPWORDS, split_sentences, words

# Typical LLM connectors/openers -> plainer alternatives ("" = drop).
CLICHES = {
    "ru": [
        (r"Важно отметить, что ", [""]),
        (r"Стоит отметить, что ", ["", "Кстати, "]),
        (r"Следует отметить, что ", [""]),
        (r"Необходимо подчеркнуть, что ", [""]),
        (r"Кроме того, ", ["А ещё ", "Плюс к этому ", "К тому же ", ""]),
        (r"Более того, ", ["Больше скажу, ", "Да и ", ""]),
        (r"Помимо этого, ", ["Ещё ", ""]),
        (r"Таким образом, ", ["Выходит, ", "Итак, ", "В итоге ", ""]),
        (r"В заключение(?: стоит сказать|,)? ", ["Если коротко, ", "В итоге ", ""]),
        (r"В целом, ", ["В общем, ", ""]),
        (r"В современном мире ", ["Сегодня ", "Сейчас "]),
        (r"играет ключевую роль", ["очень важна", "много значит"]),
        (r"играет важную роль", ["важна", "заметно влияет"]),
        (r"является неотъемлемой частью", ["давно стала частью", "входит в"]),
        (r"широкий спектр", ["целый ряд", "множество"]),
        (r"позволяет", ["даёт возможность", "помогает"]),
        (r"Однако, ", ["Но ", "Однако "]),
    ],
    "en": [
        (r"It is (?:important|worth) (?:to note|noting) that ", [""]),
        (r"It's (?:important|worth) (?:to note|noting) that ", [""]),
        (r"Moreover, ", ["Also, ", "On top of that, ", "And ", ""]),
        (r"Furthermore, ", ["Also, ", "Plus, ", ""]),
        (r"Additionally, ", ["Also, ", "And ", ""]),
        (r"In conclusion, ", ["So, ", "All in all, ", ""]),
        (r"Overall, ", ["All told, ", ""]),
        (r"Ultimately, ", ["In the end, ", ""]),
        (r"In today's (?:fast-paced |digital |modern )?world, ", ["Today, ", "These days, "]),
        (r"\bdelve into\b", ["dig into", "look at"]),
        (r"\bdelves into\b", ["digs into", "looks at"]),
        (r"\butilize\b", ["use"]),
        (r"\butilizes\b", ["uses"]),
        (r"\bleverage\b", ["use", "make use of"]),
        (r"\bplays a (?:crucial|pivotal|vital|key) role\b", ["matters a lot", "is central"]),
        (r"\ba (?:wide|broad|diverse) (?:range|array) of\b", ["many", "all sorts of"]),
        (r"\bseamless(?:ly)?\b", ["smooth"]),
        (r"\brobust\b", ["solid", "sturdy"]),
        (r"\btapestry\b", ["mix"]),
        (r"\bnavigate the complexities of\b", ["deal with"]),
    ],
}

SPLIT_POINTS = {
    "ru": [", но ", ", а ", ", и ", "; ", ", поэтому ", ", однако "],
    "en": [", but ", ", and ", "; ", ", so ", ", which means "],
}
MERGE_JOINERS = {"ru": [", и ", " — ", "; ", ", а "], "en": [", and ", " — ", "; "]}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def remove_cliches(text: str, lang: str, rnd: random.Random) -> str:
    for pat, alts in CLICHES.get(lang, []):
        def repl(m):
            r = rnd.choice(alts)
            src = m.group(0)
            # keep sentence-initial capitalization
            at_start = m.start() == 0 or text_before(m).rstrip()[-1:] in ".!?\n" or not text_before(m).strip()
            if r == "":
                return ""
            return _cap(r) if (src[:1].isupper() and at_start) else r

        def text_before(m):
            return m.string[:m.start()]

        text = re.sub(pat, repl, text)
    # capitalize sentence starts that may have been exposed by dropping an opener
    text = re.sub(r"(^|[.!?]\s+|\n)([a-zа-яё])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def _wc(s: str) -> int:
    return len(words(s))


def _can_lowercase(word: str, text: str) -> bool:
    w = word.lower()
    return w in STOPWORDS or re.search(r"(?<![\w])" + re.escape(w) + r"(?![\w])", text) is not None


def vary_lengths(text: str, lang: str, rnd: random.Random, target_cv: float = 0.5,
                 split_min: int = 18, merge_max: int = 14, rounds: int = 2) -> str:
    for _ in range(rounds):
        text = _vary_once(text, lang, rnd, target_cv, split_min, merge_max)
    return text


def _vary_once(text, lang, rnd, target_cv, split_min, merge_max):
    """Split some long sentences and merge some short neighbours until length CV reaches target."""
    out_paras = []
    for para in text.split("\n"):
        spans = split_sentences(para)
        sents = [para[a:b] for a, b in spans]
        if len(sents) < 3:
            out_paras.append(para)
            continue
        lens = [_wc(s) for s in sents]
        if st.mean(lens) == 0 or st.pstdev(lens) / st.mean(lens) >= target_cv:
            out_paras.append(para)
            continue

        res = []
        i = 0
        while i < len(sents):
            s = sents[i]
            # split long sentences at a natural clause boundary
            if _wc(s) >= split_min and rnd.random() < 0.7:
                for sp in rnd.sample(SPLIT_POINTS[lang], len(SPLIT_POINTS[lang])):
                    k = s.find(sp)
                    if k > 0 and _wc(s[:k]) >= 6 and _wc(s[k + len(sp):]) >= 6:
                        left = s[:k].rstrip(",;") + "."
                        right = s[k + len(sp):]
                        conj = sp.strip(" ,;")
                        right = _cap(conj + " " + right) if conj and conj not in ("и", "and") else _cap(right)
                        res.extend([left, right])
                        break
                else:
                    res.append(s)
                i += 1
                continue
            # merge two short neighbours
            if (i + 1 < len(sents) and _wc(s) <= merge_max and _wc(sents[i + 1]) <= merge_max
                    and s.endswith(".") and rnd.random() < 0.6):
                nxt = sents[i + 1]
                first = words(nxt)[0][2] if words(nxt) else ""
                if first and _can_lowercase(first, text):
                    j = rnd.choice(MERGE_JOINERS[lang])
                    res.append(s[:-1] + j + nxt[:1].lower() + nxt[1:])
                    i += 2
                    continue
            res.append(s)
            i += 1
        out_paras.append(" ".join(res))
    return "\n".join(out_paras)


def structure_pass(text: str, lang: str, seed: int = 0, cliches: bool = True, lengths: bool = True) -> str:
    rnd = random.Random(seed)
    if cliches:
        text = remove_cliches(text, lang, rnd)
    if lengths:
        text = vary_lengths(text, lang, rnd)
    return re.sub(r"[ \t]{2,}", " ", text)
