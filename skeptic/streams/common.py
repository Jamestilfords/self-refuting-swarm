"""Shared by the proposal streams: render a proposal file and write it under its hash."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from skeptic import api


def render(*, family: str, arm: str, rationale: str, body: str, source: str = "",
           parents=(), memory_used=(), imports: str = "") -> str:
    """Deterministic file text for one proposal. Same arguments give the same hash."""
    head = "\n".join([
        f"FAMILY = {json.dumps(family)}",
        f"ARM = {json.dumps(arm)}",
        f"RATIONALE = {json.dumps(rationale)}",
        f"SOURCE = {json.dumps(source)}",
        f"PARENTS = {json.dumps(list(parents))}",
        f"MEMORY_USED = {json.dumps(list(memory_used))}",
    ])
    pre = (imports.strip() + "\n\n\n") if imports.strip() else "\n"
    return f"{head}\n\n{pre}def score(ctx):\n{textwrap.indent(textwrap.dedent(body).strip(), '    ')}\n"


def write(out_dir, text: str, *, static: bool = True) -> str:
    """Write text to <out_dir>/<sha>.py and return the sha. Static checks run first unless disabled."""
    if static:
        api.static_check(text)
    sha = api.proposal_sha(text)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{sha}.py").write_text(text, encoding="utf-8")
    return sha


def write_manifest(out_dir, name: str, payload) -> Path:
    path = Path(out_dir) / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
