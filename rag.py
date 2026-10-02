"""Minimal grounded retrieval and answer generation for clinic FAQs."""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import get_close_matches

from ollama_client import OllamaError, chat as ollama_chat


documents = [
    "Appointments can be cancelled up to 24 hours before the scheduled time.",
    "Dr. Ahmed is available Monday, Wednesday, and Friday.",
    "Patients can reschedule appointments through the patient portal.",
    "Insurance claims are usually processed within 7 business days.",
    "Emergency appointments should be escalated to a human operator.",
]

_STOP_WORDS = {
    "a", "an", "and", "are", "be", "can", "do", "for", "how", "i", "in", "is",
    "of", "on", "the", "to", "was", "what", "when", "where", "with", "you",
}
_CLARIFICATION_TERMS = {
    "appointment", "cancel", "claim", "doctor", "emergency", "insurance",
    "name", "patient", "portal", "reschedule", "schedule",
}


def _normalise_token(token: str) -> str:
    """Apply a few deliberate forms needed by the small clinic corpus."""
    if token == "doctor":
        return "dr"
    if token == "cancellation":
        return "cancel"
    if token.endswith("led") and len(token) > 5:
        return token[:-3]  # cancelled -> cancel
    if token.endswith("s") and len(token) > 3:
        return token[:-1]  # appointments -> appointment
    return token


def _spelling_clarification(query: str) -> str | None:
    """Request confirmation for likely domain-word misspellings, never autocorrect."""
    for token in re.findall(r"[a-z0-9]+", query.lower()):
        if token in _STOP_WORDS or token in _CLARIFICATION_TERMS or _normalise_token(token) in _CLARIFICATION_TERMS:
            continue
        suggestion = get_close_matches(token, _CLARIFICATION_TERMS, n=1, cutoff=0.75)
        if suggestion:
            return f"I may have misunderstood '{token}'. Did you mean '{suggestion[0]}'? Please clarify."
    return None


def _tokens(text: str) -> list[str]:
    return [
        _normalise_token(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOP_WORDS
    ]


def _ranked_documents(query: str, corpus: list[str]) -> list[tuple[float, str]]:
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    document_tokens = [_tokens(document) for document in corpus]
    document_frequency = Counter(
        token for tokens in document_tokens for token in set(tokens)
    )
    total_documents = len(corpus)
    ranked: list[tuple[float, str]] = []
    for document, tokens in zip(corpus, document_tokens):
        counts = Counter(tokens)
        score = sum(
            counts[token] * math.log((total_documents + 1) / (document_frequency[token] + 1))
            for token in set(query_tokens)
        )
        if score > 0:
            ranked.append((score, document))
    return sorted(ranked, key=lambda item: item[0], reverse=True)


def retrieve(query: str, documents: list[str], top_k: int = 2) -> list[str]:
    """Return at most ``top_k`` keyword/IDF-ranked knowledge-base chunks."""
    if top_k <= 0:
        return []
    return [document for _, document in _ranked_documents(query, documents)[:top_k]]


def _answer_with_ollama(query: str, context: list[str]) -> str:
    """Generate an answer locally, constrained to the retrieved context."""
    response = ollama_chat(
        [
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied context. If it does not answer the "
                    "question, say that the information is unavailable. Do not add facts."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n- " + "\n- ".join(context) + f"\n\nQuestion: {query}",
            },
        ]
    )
    answer = response["message"].get("content", "").strip()
    if not answer:
        raise OllamaError("Ollama returned an empty answer.")
    return answer


def answer_with_context(query: str) -> dict[str, object]:
    """Answer only when retrieval finds relevant source material."""
    clarification = _spelling_clarification(query)
    if clarification:
        return {"success": False, "answer": clarification, "sources": []}

    ranked = _ranked_documents(query, documents)
    sources = [document for _, document in ranked[:2]]
    if not sources:
        return {
            "success": False,
            "answer": "I do not have enough information in the knowledge base to answer that.",
            "sources": [],
        }
    try:
        answer = _answer_with_ollama(query, sources)
    except OllamaError:
        return {
            "success": False,
            "answer": "The relevant sources were found, but the local LLM is unavailable.",
            "sources": sources,
        }
    return {"success": True, "answer": answer, "sources": sources}
