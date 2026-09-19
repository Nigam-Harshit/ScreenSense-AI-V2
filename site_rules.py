"""
site_rules.py  --  Deterministic rule layer for website opening, granular scrolling,
                   and fail-safe handling of unsupported on-screen content actions.
"""

import re

# Fixed alias dictionary mapping exact keys to canonical HTTPS URLs.
# Only URLs defined in this dictionary are ever opened.
SITE_ALIASES = {
    "youtube": "https://www.youtube.com",
    "you tube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "apple": "https://www.apple.com",
    "github": "https://github.com",
    "linkedin": "https://www.linkedin.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
    "kaggle": "https://kaggle.com",
    "huggingface": "https://huggingface.co",
    "hugging face": "https://huggingface.co",
}

# Granular scroll pixel amounts: chosen, pending tuning
SCROLL_AMOUNTS = {
    "bit": 300,
    "little": 300,
    "slightly": 300,
    "tiny": 300,
    "lot": 1600,
    "far": 1600,
    "much": 1600,
}

UNSUPPORTED_MESSAGE = (
    "Not supported yet: clicking on on-screen content. "
    "Supported: open a site, scroll, window and system commands."
)

_OPEN_VERBS = [
    "take me to",
    "browse to",
    "click on",
    "tap on",
    "open up",
    "fire up",
    "start up",
    "pull up",
    "go to",
    "goto",
    "launch",
    "visit",
    "click",
    "tap",
    "open",
]

_LEADING_FILLERS = [
    "could you please",
    "would you please",
    "can you please",
    "could you",
    "would you",
    "can you",
    "please",
    "hey",
]

_TRAILING_FILLERS = [
    "for me",
    "please",
    "now",
]

_TRAILING_SITE_NOUNS = [
    "home page",
    "homepage",
    "web page",
    "webpage",
    "website",
    "site",
    "page",
]

_EXCEPTION_WORDS = frozenset({
    "close", "minimize", "minimise", "maximize", "maximise",
    "button", "window", "play", "pause"
})

_ORDINALS = r"(?:1st|2nd|3rd|4th|5th|first|second|third|fourth|fifth|top)"
_CONTENT_NOUNS = r"(?:website|site|result|link|video|page)"
_UNSUPPORTED_ORDINAL_RE = re.compile(
    rf"^(?:click\s+on|click|tap\s+on|tap|open|select)\b.*?\b{_ORDINALS}\b.*?\b{_CONTENT_NOUNS}\b",
    re.IGNORECASE
)


def _normalize(command: str) -> str:
    """Normalize input command: lowercase, trim, replace typographic apostrophes, drop punctuation."""
    s = command.lower().strip()
    s = s.replace("’", "'")
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_fillers(norm: str) -> str:
    """Strip optional leading and trailing conversational fillers."""
    s = norm.strip()
    changed = True
    while changed:
        changed = False
        for filler in _LEADING_FILLERS:
            if s == filler:
                return ""
            if s.startswith(filler + " "):
                s = s[len(filler):].strip()
                changed = True
                break
        for filler in _TRAILING_FILLERS:
            if s == filler:
                return ""
            if s.endswith(" " + filler):
                s = s[:-len(" " + filler)].strip()
                changed = True
                break
    return s


def match_rule(command: str):
    """
    Pure function that evaluates a natural-language command against deterministic rules.

    Returns:
      None                               if no rule matches (command falls through to classifier)
      ("open_website", url)              for recognized site aliases
      ("scroll", direction, amount)      for granular scroll commands with quantity words
      ("unsupported", reason)            for unsupported on-screen content interactions
    """
    norm = _normalize(command)
    if not norm:
        return None

    norm = _strip_fillers(norm)
    if not norm:
        return None

    # ── Rule 1: Open website ──────────────────────────────────────────────────
    for verb in _OPEN_VERBS:
        if norm.startswith(verb):
            rest = norm[len(verb):].strip()
            # Optional "the" after verb
            if rest.startswith("the "):
                rest = rest[4:].strip()

            # Optional trailing site noun
            for noun in _TRAILING_SITE_NOUNS:
                if rest.endswith(" " + noun):
                    rest = rest[:-len(" " + noun)].strip()
                    break

            # Handle possessive ('s, ' s, or trailing s like apples / apple s)
            candidate = rest
            if candidate.endswith("'s"):
                candidate = candidate[:-2].strip()
            elif candidate.endswith("' s"):
                candidate = candidate[:-3].strip()
            elif candidate.endswith(" s") and not candidate.endswith("pass"):
                candidate = candidate[:-2].strip()

            if candidate in SITE_ALIASES:
                return ("open_website", SITE_ALIASES[candidate])

            # Check trailing s (e.g. "apples") if candidate without 's' matches alias
            if candidate.endswith("s") and len(candidate) > 2 and candidate[:-1] in SITE_ALIASES:
                return ("open_website", SITE_ALIASES[candidate[:-1]])

    # ── Rule 2: Scroll with quantity word ─────────────────────────────────────
    m_scroll = re.match(r"^scroll\s+(up|down)\b(.*)$", norm)
    if m_scroll:
        direction = m_scroll.group(1)
        rest = m_scroll.group(2).strip()
        tokens = rest.split()
        amount = None
        for q, amt in SCROLL_AMOUNTS.items():
            if q in tokens:
                amount = amt
                break
        if amount is not None:
            return ("scroll", direction, amount)
        elif not rest:
            # Plain scroll down / scroll up with no quantity word returns None
            return None

    # ── Rule 3: Unsupported on-screen content interaction ─────────────────────
    # (b) Leading click / tap / open / select verb, ordinal, and content noun
    if _UNSUPPORTED_ORDINAL_RE.search(norm):
        return ("unsupported", UNSUPPORTED_MESSAGE)

    # (a) Leading click or tap verb, no alias match, and no exception words
    click_tap_verbs = ["click on", "click", "tap on", "tap"]
    is_click_tap = any(norm.startswith(v) for v in click_tap_verbs)
    if is_click_tap:
        tokens = set(re.findall(r"\b\w+\b", norm))
        if not (tokens & _EXCEPTION_WORDS):
            return ("unsupported", UNSUPPORTED_MESSAGE)

    return None

