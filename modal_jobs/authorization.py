"""Workspace-scoped run receipts issued after a local verify_published_freeze."""

from skeptic.store import GuardError, UnsealPermit


def freeze_receipt(permit: UnsealPermit, purpose: str, **extra) -> dict:
    """{"commit", "digest", "purpose", **extra}. Called by the local coordinator only."""
    return {"commit": permit.commit, "digest": permit.digest, "purpose": purpose, **extra}


def accept_freeze_receipt(authorization, commit: str, digest: str, purpose: str) -> UnsealPermit:
    """Return UnsealPermit(digest, commit) if authorization["freeze"] matches all three.

    Raise GuardError("Invalid run freeze authorization") on anything else:
    a missing key, a non-dict, or a wrong commit, digest or purpose.
    """
    try:
        receipt = authorization.get("freeze")
    except (AttributeError, TypeError):
        raise GuardError("Invalid run freeze authorization") from None
    if (not isinstance(receipt, dict)
            or receipt.get("commit") != commit
            or receipt.get("digest") != digest
            or receipt.get("purpose") != purpose):
        raise GuardError("Invalid run freeze authorization")
    return UnsealPermit(digest, commit)
