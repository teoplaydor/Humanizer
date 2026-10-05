import re

WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
# Sentence end: . ! ? … followed by whitespace and an uppercase letter / quote / digit.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])[\"»)]?\s+(?=[\"«(]?[A-ZА-ЯЁ0-9])")

STOPWORDS = set("""
the a an and or but if then than that this these those there their they them is are was were be been being
have has had do does did of in on at to for from by with as into over under about after before between
not no nor so too very can could should would will may might must it its we our you your he she his her
which who whom whose what when where why how also such more most other some any each all both
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было
вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь
опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам
чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь
этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой
хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою
этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это также которые который
которая которое является являются
""".split())


def detect_lang(text: str) -> str:
    cyr = sum(1 for c in text if "а" <= c.lower() <= "я" or c.lower() == "ё")
    lat = sum(1 for c in text if "a" <= c.lower() <= "z")
    return "ru" if cyr >= lat else "en"


def split_sentences(text: str):
    """Return list of (start, end) spans of sentences in text (paragraph breaks also split)."""
    spans = []
    for pm in re.finditer(r"[^\n]+", text):
        para, base = pm.group(0), pm.start()
        pos = 0
        for m in SENT_SPLIT_RE.finditer(para):
            seg = para[pos:m.start()]
            if seg.strip():
                spans.append((base + pos, base + m.start()))
            pos = m.end()
        if para[pos:].strip():
            spans.append((base + pos, base + len(para)))
    # trim whitespace
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def words(text: str):
    return [(m.start(), m.end(), m.group(0)) for m in WORD_RE.finditer(text)]


def script_of(word: str) -> str:
    w = word.lower()
    if any("а" <= c <= "я" or c == "ё" for c in w):
        return "cyr"
    if any("a" <= c <= "z" for c in w):
        return "lat"
    return "other"


def match_case(src: str, new: str) -> str:
    if src.isupper() and len(src) > 1:
        return new.upper()
    if src[:1].isupper():
        return new[:1].upper() + new[1:]
    return new
