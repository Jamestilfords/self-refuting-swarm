"""Offline tests for the shared freeze receipt. No GitHub or Modal calls."""

import pytest

from modal_jobs.authorization import accept_freeze_receipt, freeze_receipt
from skeptic.store import GuardError, UnsealPermit

PERMIT = UnsealPermit("a" * 64, "b" * 40)


def test_matching_receipt_returns_the_permit():
    authorization = {"freeze": freeze_receipt(PERMIT, "seal", targets=[("clinc150/bert", "c" * 12)])}
    assert accept_freeze_receipt(authorization, PERMIT.commit, PERMIT.digest, "seal") == PERMIT


@pytest.mark.parametrize("authorization", [
    None,
    [],
    {},
    {"other": True},
    {"freeze": ["not", "a", "dict"]},
    {"freeze": freeze_receipt(UnsealPermit(PERMIT.digest, "c" * 40), "plain_ece")},
    {"freeze": freeze_receipt(UnsealPermit("d" * 64, PERMIT.commit), "plain_ece")},
    {"freeze": freeze_receipt(PERMIT, "unblind")},
])
def test_bad_receipt_is_refused(authorization):
    with pytest.raises(GuardError, match="Invalid run freeze authorization"):
        accept_freeze_receipt(authorization, PERMIT.commit, PERMIT.digest, "plain_ece")
