"""L4: shared memory for the agents: source cards, proposal cards, verdict summaries and lessons.

Owner: @Jamestilfords (research engineer).

Four agents write (Scout, Proposer, Critic, Analyst) and two read (Proposer, Analyst). Every entry gets
an id such as m_0042, and a proposal lists the ids it used in MEMORY_USED. Sealed test scores never
enter memory: `write` refuses any entry that mentions a test split.

What lives where (sponsor requirement):
    runs/memory.jsonl   the local record: every entry with its id. Ids in MEMORY_USED resolve here, and
                        the file replays offline.
    past.dev            every entry is also ingested there, and agents query it with `recall` to get
                        ranked, dated evidence across everything any agent wrote.

    LocalMemory     the local record only. Used when no past.dev key is set.
    PastDevMemory   LocalMemory plus ingest and recall against https://api.past.dev.

    python -m skeptic.memory --ping     one ingest and one recall, printing the raw responses
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

KINDS = ("source", "proposal", "verdict", "lesson")
AGENTS = ("scout", "proposer", "critic", "analyst")
_SEALED_WORDS = ("oos_test", "test__", "sealed")


class LocalMemory:
    def __init__(self, path="runs/memory.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
                         if line.strip()] if self.path.exists() else []

    def write(self, kind: str, author: str, content: dict) -> str:
        """Append one entry and return its id."""
        if kind not in KINDS or author not in AGENTS:
            raise ValueError(f"kind must be in {KINDS} and author in {AGENTS}")
        blob = json.dumps(content, sort_keys=True)
        if any(w in blob.lower() for w in _SEALED_WORDS):
            raise ValueError("memory entries must not mention sealed test data")
        entry = {"id": f"m_{len(self._entries) + 1:04d}", "kind": kind, "author": author,
                 "time": round(time.time(), 1), "content": content}
        self._entries.append(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry["id"]

    def read(self, kinds=None, limit: int | None = None, reader: str | None = None) -> list[dict]:
        """Newest entries last. `kinds` filters by kind; `limit` keeps the most recent per kind."""
        out = []
        for kind in (kinds or KINDS):
            rows = [e for e in self._entries if e["kind"] == kind]
            out += rows[-limit:] if limit else rows
        return sorted(out, key=lambda e: e["id"])

    def get(self, entry_id: str) -> dict | None:
        return next((e for e in self._entries if e["id"] == entry_id), None)

    def ids(self) -> set[str]:
        return {e["id"] for e in self._entries}

    def stats(self) -> dict:
        """Who wrote what: the evidence for 'more than one agent writes'."""
        out: dict = {}
        for e in self._entries:
            out.setdefault(e["author"], {}).setdefault(e["kind"], 0)
            out[e["author"]][e["kind"]] += 1
        return out


def entry_text(e: dict) -> str:
    """One entry as plain text, for past.dev and for prompts."""
    c = e["content"]
    if e["kind"] == "source":
        return f"Source {c.get('title', '')} ({c.get('url', '')}) describes: {c.get('method', '')}"
    if e["kind"] == "proposal":
        return f"Proposal {c.get('sha', '')} [{c.get('family', '')}/{c.get('arm', '')}]: {c.get('rationale', '')}"
    if e["kind"] == "verdict":
        return f"Verdict for proposal {c.get('sha', '')}: {c.get('verdict', '')}. {c.get('summary', '')}"
    return f"Lesson: {c.get('text', '')}"


class PastDevMemory(LocalMemory):
    """LocalMemory that also ingests every entry into past.dev and can recall ranked evidence from it.

    A past.dev failure never stops the loop: the local record is written first, and errors are counted
    in `self.errors` and printed once.
    """

    BASE = "https://api.past.dev/api/v1"

    def __init__(self, path="runs/memory.jsonl", api_key: str | None = None, audience: str | None = None,
                 timeout: float = 20.0, namespace: str = ""):
        """No audience by default: past.dev then treats each entry as project-visible, readable by every
        identity, which is what a memory shared by all agents needs."""
        super().__init__(path)
        self.key = api_key or os.environ.get("PASTDEV_API_KEY") or os.environ.get("PAST_API_KEY")
        if not self.key:
            raise ValueError("set PASTDEV_API_KEY in .env")
        self.audience, self.timeout = audience, timeout
        self.namespace = namespace
        self.errors, self.ingested, self.recalls, self._warned = 0, 0, 0, False

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(self.BASE + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            raw = r.read().decode()
        return json.loads(raw) if raw.strip() else {}

    def _safe(self, method: str, path: str, body: dict | None = None) -> dict | None:
        try:
            return self._call(method, path, body)
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            self.errors += 1
            if not self._warned:
                print(f"[past.dev] {method} {path} failed: {type(e).__name__}; local memory retained")
                self._warned = True
            return None

    def write(self, kind: str, author: str, content: dict) -> str:
        entry_id = super().write(kind, author, content)
        e = self.get(entry_id)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(e["time"]))
        remote_id = f"{self.namespace}:{entry_id}" if self.namespace else entry_id
        references = f"[{remote_id}] [{entry_id}]" if self.namespace else f"[{entry_id}]"
        body = {"id": remote_id, "content": f"{references} {entry_text(e)}",
                "label": f"{kind} written by the {author} agent", "timestamp": stamp}
        if self.audience:
            body["audience"] = self.audience
        out = self._safe("POST", "/ingest", body)
        self.ingested += out is not None
        return entry_id

    def recall(self, query: str, reader: str = "proposer", level: str = "medium") -> list[dict]:
        """Ranked evidence from past.dev. Each item: {"text", "ids"}; ids are the memory ids found in it."""
        query = f"Within run {self.namespace}: {query}" if self.namespace else query
        out = self._safe("POST", "/recall", {"query": query, "identity": reader, "level": level})
        if not out:
            return []
        self.recalls += 1
        known, items = self.ids(), []
        for r in out.get("results", []):
            text = r.get("content") or ""
            blob = text + " " + json.dumps(r.get("sources", "")) + " " + json.dumps(r.get("artifact", ""))
            if self.namespace and not any(f"{self.namespace}:{entry_id}" in blob for entry_id in known):
                continue
            if any(w in text.lower() for w in _SEALED_WORDS):
                continue
            ids = sorted({w.strip("[](),.:\"'") for w in blob.replace('"', " ").split() if w.strip("[](),.:\"'") in known})
            items.append({"text": " ".join(text.split())[:400], "ids": ids})
        return items


def get_memory(path="runs/memory.jsonl", *, backend="auto", namespace=""):
    """PastDevMemory when a key is set, otherwise LocalMemory."""
    if backend not in ("auto", "local", "pastdev"):
        raise ValueError("Unknown memory backend")
    if backend == "pastdev" or (backend == "auto" and (os.environ.get("PASTDEV_API_KEY") or os.environ.get("PAST_API_KEY"))):
        return PastDevMemory(path, namespace=namespace)
    return LocalMemory(path)


def format_for_prompt(entries: list[dict], max_chars: int = 6000) -> str:
    """Compact text for an LLM prompt: one line per entry, newest kept when over budget."""
    lines = []
    for e in entries:
        c = e["content"]
        if e["kind"] == "source":
            text = f"{c.get('title', '')} ({c.get('url', '')}): {c.get('method', '')}"
        elif e["kind"] == "proposal":
            text = f"[{c.get('family', '')}/{c.get('arm', '')}] {c.get('rationale', '')}"
        elif e["kind"] == "verdict":
            text = f"{c.get('sha', '')} -> {c.get('verdict', '')}: {c.get('summary', '')}"
        else:
            text = c.get("text", "")
        lines.append(f"{e['id']} {e['kind']}: {text}".strip())
    while lines and sum(len(x) + 1 for x in lines) > max_chars:
        lines.pop(0)
    return "\n".join(lines) if lines else "(memory is empty)"


def _ping() -> None:
    """One ingest and one recall against past.dev, printing raw responses. Uses a scratch local file."""
    import tempfile
    from dotenv import load_dotenv
    load_dotenv()
    m = PastDevMemory(Path(tempfile.mkdtemp()) / "ping.jsonl")
    body = {"id": f"ping-{int(time.time())}", "label": "connection test",
            "content": "Connection test: the Scout agent found that k-nearest-neighbour distance detects far out-of-scope queries.",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = m._safe("POST", "/ingest", body)
    print("INGEST ->", json.dumps(out, indent=2))
    ing = (out or {}).get("ingestionId") or (out or {}).get("id")
    for _ in range(15):
        if not ing:
            break
        time.sleep(2)
        status = m._safe("GET", f"/ingest/{ing}")
        print("STATUS ->", json.dumps(status))
        if (status or {}).get("status") in ("completed", "failed"):
            break
    print("RECALL ->", json.dumps(m._safe("POST", "/recall", {"query": "what detects far out-of-scope queries?",
                                                              "identity": "proposer", "level": "medium"}), indent=2)[:3000])


if __name__ == "__main__":
    import sys
    if "--ping" in sys.argv:
        _ping()
    else:
        print(__doc__)
