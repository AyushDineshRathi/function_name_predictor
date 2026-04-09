import re
from typing import Iterable


VERB_NORMALIZATION = {
    "adds": "add",
    "adding": "add",
    "converts": "convert",
    "converting": "convert",
    "returns": "return",
}

REDUNDANT_WORDS = {
    "function",
    "functions",
    "method",
    "methods",
    "return",
    "returns",
}

TYPE_HINTS = {
    "int",
    "integer",
    "float",
    "double",
    "string",
    "char",
    "boolean",
    "bool",
    "long",
    "short",
    "byte",
    "list",
    "array",
    "map",
    "set",
    "dict",
    "object",
    "file",
}


def _collapse_repeated_halves(token: str) -> str:
    # Handles artifacts like "datadata" -> "data".
    if len(token) % 2 == 0:
        mid = len(token) // 2
        if token[:mid] == token[mid:]:
            return token[:mid]
    return token


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []
    return cleaned.split()


def normalize_tokens(
    text: str,
    remove_redundant: bool = True,
    dedupe: bool = True,
) -> str:
    if text is None:
        return ""

    tokens = _tokenize(str(text))
    normalized: list[str] = []
    seen = set()

    for raw_token in tokens:
        token = _collapse_repeated_halves(raw_token)
        token = VERB_NORMALIZATION.get(token, token)

        if remove_redundant and token in REDUNDANT_WORDS:
            continue

        if dedupe:
            if token in seen:
                continue
            seen.add(token)

        normalized.append(token)

    return " ".join(normalized)


def normalize_params(params_text: str) -> str:
    if params_text is None:
        return ""

    parts = [p.strip() for p in str(params_text).split(",") if p.strip()]
    param_types: list[str] = []

    for part in parts:
        tokens = _tokenize(part)
        if not tokens:
            continue

        # If the tail token looks like an identifier and there are prior tokens,
        # treat it as variable name and keep type-ish tokens.
        if len(tokens) > 1 and tokens[-1] not in TYPE_HINTS:
            tokens = tokens[:-1]

        # Keep only informative tokens (types / shape hints), preserving order.
        kept = [t for t in tokens if t in TYPE_HINTS]
        if not kept:
            kept = tokens

        param_types.extend(kept)

    # Keep duplicates in params (e.g., "int int") to preserve count signal.
    return " ".join(param_types)


def _join_unique_tokens(chunks: Iterable[str]) -> str:
    seen = set()
    out: list[str] = []
    for chunk in chunks:
        for token in _tokenize(chunk):
            token = VERB_NORMALIZATION.get(_collapse_repeated_halves(token), token)
            if token in REDUNDANT_WORDS:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
    return " ".join(out)


def build_structured_metadata(
    description: str,
    parameters: str,
    return_type: str,
    keywords: str,
) -> str:
    desc = normalize_tokens(description, remove_redundant=True, dedupe=True)
    params = normalize_params(parameters)
    ret = normalize_tokens(return_type, remove_redundant=True, dedupe=True)
    kw = _join_unique_tokens([keywords])
    return f"desc:{desc} | params:{params} | return:{ret} | keywords:{kw}"


def build_structured_metadata_from_text(metadata_text: str) -> str:
    text = str(metadata_text or "")

    if "|" in text and any(prefix in text.lower() for prefix in ("desc:", "params:", "return:", "keywords:")):
        fields = {"desc": "", "params": "", "return": "", "keywords": ""}
        for part in text.split("|"):
            part = part.strip()
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key = key.strip().lower()
            if key in fields:
                fields[key] = value.strip()
        return build_structured_metadata(
            description=fields["desc"],
            parameters=fields["params"],
            return_type=fields["return"],
            keywords=fields["keywords"],
        )

    lower = text.lower()
    keyword_split = re.split(r"\bkeywords?\b", lower, maxsplit=1)
    before_keywords = keyword_split[0].strip()
    keyword_text = keyword_split[1].strip() if len(keyword_split) > 1 else ""

    return_split = re.split(r"\breturn\b", before_keywords, maxsplit=1)
    desc_text = return_split[0].strip()
    return_text = return_split[1].strip().split()[0] if len(return_split) > 1 and return_split[1].strip() else ""

    return build_structured_metadata(
        description=desc_text,
        parameters="",
        return_type=return_text,
        keywords=keyword_text,
    )
