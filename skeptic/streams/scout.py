"""L5: the Scout. Searches the web with Exa for scoring methods and writes source cards to memory.

Each card carries its url, title, date, a one- or two-sentence description of the method, and the
failure it targets. The Proposer may only cite a SOURCE that appears on a source card, so every
citation in a proposal traces back to a real Exa result.
"""
from __future__ import annotations

from collections import Counter

DEFAULT_FOCI = [
    "out-of-scope intent detection with Mahalanobis distance on BERT features",
    "energy-based out-of-distribution detection from classifier logits",
    "combining feature-space density with logit scores for OOD detection",
    "ViM virtual-logit matching residual space out-of-distribution score",
    "k-nearest-neighbour out-of-distribution detection on normalized embeddings",
    "near-OOD versus far-OOD detection which feature layer works best",
    "relative Mahalanobis distance and Mahalanobis++ feature normalization",
    "post-hoc OOD detection methods for text classifiers without retraining",
]
SUMMARY_SYSTEM = ("You read a web page about out-of-distribution or out-of-scope detection. In at most two "
                  "sentences, state the scoring method it describes in a form someone could implement with numpy "
                  "on cached classifier logits and feature vectors. If the page describes no such method, reply NONE.")


def top_rejection(memory) -> str | None:
    """The most common non-accept verdict in memory, e.g. 'INERT', or None when there are no verdicts yet."""
    counts = Counter(e["content"].get("verdict") for e in memory.read(["verdict"])
                     if e["content"].get("verdict") not in (None, "ACCEPT"))
    return counts.most_common(1)[0][0] if counts else None


def _exa_results(exa, query: str, k: int):
    """Exa call site (sponsor requirement). Tries the content-returning calls first."""
    for call in (lambda: exa.search_and_contents(query, num_results=k, text={"max_characters": 2500}),
                 lambda: exa.search(query, num_results=k, contents={"text": {"max_characters": 2500}}),
                 lambda: exa.search(query, num_results=k)):
        try:
            return list(call().results)
        except (TypeError, AttributeError):
            continue
    return []


class Scout:
    def __init__(self, memory, llm, exa, num_results: int = 3):
        self.memory, self.llm, self.exa, self.k = memory, llm, exa, num_results
        self._round = 0
        self._skipped: set[str] = set()     # urls with no usable method; not summarized twice
        self.usage = {"input_tokens": 0, "output_tokens": 0, "searches": 0}

    def search(self) -> list[str]:
        """One Exa search. Returns the ids of the source cards written."""
        focus = DEFAULT_FOCI[self._round % len(DEFAULT_FOCI)]
        failure = top_rejection(self.memory)
        query = f"{focus} (avoid changes that are {failure.lower()})" if failure else focus
        self._round += 1
        self.usage["searches"] += 1
        seen = {e["content"].get("url") for e in self.memory.read(["source"])}
        ids = []
        for r in _exa_results(self.exa, query, self.k):
            url, text = getattr(r, "url", None), (getattr(r, "text", None) or "")
            if not url or url in seen or url in self._skipped:
                continue
            title = getattr(r, "title", None) or url
            method, usage = self.llm(SUMMARY_SYSTEM, f"Title: {title}\n\n{text[:2500]}" if text else f"Title: {title}")
            for key in ("input_tokens", "output_tokens"):
                self.usage[key] += usage.get(key, 0)
            if method.strip().upper().startswith("NONE"):
                self._skipped.add(url)
                continue
            seen.add(url)
            ids.append(self.memory.write("source", "scout", {
                "url": url, "title": title, "date": getattr(r, "published_date", None),
                "method": method.strip(), "target_failure": failure or "none yet", "query": query}))
        return ids
