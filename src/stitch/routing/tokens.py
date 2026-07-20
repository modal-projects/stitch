"""Prompt token-id extraction for routing decisions.

The router's prefix-cache awareness (the gorgo radix trie) keys on token-id
sequences, ideally matching the engine's own tokenization so cached-prefix
estimates line up with SGLang's RadixAttention. Rollout traffic arrives in
three shapes, handled in fidelity order:

1. ``/generate`` with ``input_ids`` — already tokenized, use as-is.
2. ``/generate`` with ``text``, or chat/completions with a tokenizer
   available — tokenize the way the server would (``apply_chat_template``
   with ``add_generation_prompt=True`` for chat).
3. No tokenizer — fall back to the prompt's UTF-8 bytes as pseudo-token ids.
   Byte values (< 256) fit the trie's uint32 edges, and prefix relationships
   are preserved, so routing stays cache-aware in byte units; only the
   token *counts* are inflated (~4x), which the gorgo weights absorb. The
   fallback is self-consistent as long as every request goes through it.

The tokenizer is injected (any object with ``encode`` and, for chat,
``apply_chat_template``) so this module stays dependency-free; deployments
pass a HuggingFace tokenizer, tests pass a fake.
"""

from __future__ import annotations

from typing import Any

# Routes whose bodies carry a prompt worth routing on. Matches the sidecar's
# VERSIONED_ROUTES (service.py) — the same requests that are version-gated
# are the ones worth load-balancing.
ROUTED_PATHS = ("generate", "v1/chat/completions", "v1/completions")


def _message_text(content: Any) -> str:
    """Extract the text payload from an OpenAI chat-completions ``content``
    field (a string, or a list of content blocks). Non-text blocks are
    dropped — they don't contribute to the routing token count."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "") or ""
                    if text:
                        parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _bytes_as_tokens(text: str) -> list[int]:
    return list(text.encode("utf-8", "surrogatepass"))


def _chat_text(payload: dict) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        _message_text(m.get("content")) for m in messages if isinstance(m, dict)
    )


def extract_token_ids(
    path: str,
    payload: dict | None,
    tokenizer: Any | None,
) -> tuple[list[int], int]:
    """Return ``(token_ids, request_tokens)`` for a rollout request body.

    ``token_ids`` is the sequence the radix trie keys on (for a batched
    ``input_ids`` request, the first sequence — the whole batch lands on one
    replica, and one sequence is enough for a prefix estimate).
    ``request_tokens`` is the load-counter size: the total prompt tokens the
    chosen replica will prefill (the batch sum). Returns ``([], 0)`` for
    non-routed paths, empty bodies, or tokenization failures — the policy
    then routes load-only, which is the correct degradation.
    """
    route = (path or "").strip("/")
    if route not in ROUTED_PATHS or not isinstance(payload, dict):
        return [], 0

    if route == "generate":
        ids = payload.get("input_ids")
        if isinstance(ids, list) and ids:
            if isinstance(ids[0], list):  # batched: list[list[int]]
                seqs = [s for s in ids if isinstance(s, list)]
                first = [t for t in (seqs[0] if seqs else []) if isinstance(t, int)]
                total = sum(len(s) for s in seqs)
                return first, total
            if all(isinstance(t, int) for t in ids):
                return list(ids), len(ids)
        text = payload.get("text")
        if isinstance(text, list):  # batched text
            texts = [t for t in text if isinstance(t, str)]
            if not texts:
                return [], 0
            if tokenizer is not None:
                try:
                    encoded = [tokenizer.encode(t) for t in texts]
                    return list(encoded[0]), sum(len(e) for e in encoded)
                except Exception:
                    return [], 0
            byte_seqs = [_bytes_as_tokens(t) for t in texts]
            return byte_seqs[0], sum(len(s) for s in byte_seqs)
        if isinstance(text, str) and text:
            if tokenizer is not None:
                try:
                    ids = list(tokenizer.encode(text))
                    return ids, len(ids)
                except Exception:
                    return [], 0
            ids = _bytes_as_tokens(text)
            return ids, len(ids)
        return [], 0

    if route == "v1/chat/completions":
        messages = payload.get("messages")
        if tokenizer is not None and isinstance(messages, list) and messages:
            # apply_chat_template(..., add_generation_prompt=True) matches
            # SGLang's server-side rendering, so counts and ids line up
            # with what its KV cache actually holds.
            try:
                rendered = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                ids = list(tokenizer.encode(rendered))
                return ids, len(ids)
            except Exception:
                return [], 0
        text = _chat_text(payload)
        if not text:
            return [], 0
        ids = _bytes_as_tokens(text)
        return ids, len(ids)

    # v1/completions: a raw prompt string (or list of strings).
    prompt = payload.get("prompt")
    if isinstance(prompt, list):
        prompt = "\n".join(p for p in prompt if isinstance(p, str))
    if not isinstance(prompt, str) or not prompt:
        return [], 0
    if tokenizer is not None:
        try:
            ids = list(tokenizer.encode(prompt))
            return ids, len(ids)
        except Exception:
            return [], 0
    ids = _bytes_as_tokens(prompt)
    return ids, len(ids)
