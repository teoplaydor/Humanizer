"""LLM rewrite pass through the Claude API (needs ANTHROPIC_API_KEY or another configured credential)."""
import os

MODEL = os.environ.get("HUMANIZER_CLAUDE_MODEL", "claude-opus-5-5")

SYSTEM = """You rewrite text so that it reads like it was written by a person, while keeping its meaning.

Hard rules:
- Keep every fact, number, name, claim and the overall argument. Add no new facts, drop nothing substantive.
- Keep the language of the input ({lang_name}) and roughly the same length (±15%).
- Output only the rewritten text, no preface or commentary.

How to make it read as human-written:
- Vary sentence length a lot: mix a few very short sentences with longer, more winding ones.
- Avoid stock connectors and filler typical of AI text ({cliches}).
- Prefer specific, slightly less obvious word choices over the most expected ones, but stay natural for the register; never use rare or archaic words just to be unusual.
- Vary sentence openings and structure; don't start consecutive sentences the same way; avoid neat parallel triplets and symmetric lists.
- An occasional aside, parenthetical, or rhetorical question is fine where it fits the register.
"""

FOCUS = """The following sentences were flagged as the most statistically predictable. Rewrite only these sentences \
(change wording and structure, keep meaning), and leave the rest of the text exactly as is:
{items}
"""

CLICHE_HINT = {
    "ru": "«Важно отметить», «Кроме того», «Таким образом», «В заключение», «играет ключевую роль», «в современном мире»",
    "en": "'Moreover', 'Furthermore', 'It is important to note', 'In conclusion', 'delve', 'plays a crucial role', 'tapestry'",
}
LANG_NAME = {"ru": "Russian", "en": "English"}


def available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def rewrite(text: str, lang: str, focus_sentences=None, effort: str = "medium") -> str:
    import anthropic

    client = anthropic.Anthropic()
    system = SYSTEM.format(lang_name=LANG_NAME.get(lang, lang), cliches=CLICHE_HINT.get(lang, ""))
    user = text
    if focus_sentences:
        items = "\n".join(f"- {s}" for s in focus_sentences)
        user = FOCUS.format(items=items) + "\n\nFull text:\n" + text
    # server-side fallback: if the primary model declines, the API retries on a suitable fallback model
    with client.beta.messages.stream(
        model=MODEL,
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},
        max_tokens=32000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort},
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("Claude declined to rewrite this text")
    out = "".join(b.text for b in msg.content if b.type == "text").strip()
    return out or text
