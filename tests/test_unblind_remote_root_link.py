import pytest

from modal_jobs.unblind_remote import hash_sealed_files
from skeptic.store import GuardError


def test_root_that_is_a_link_passes(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "plain_ece.json").write_text("{}")
    link = tmp_path / "sealed"
    link.symlink_to(real, target_is_directory=True)
    result = hash_sealed_files(link, "c", "d", ["plain_ece.json"])
    assert list(result["files"]) == ["plain_ece.json"]
    assert result["missing"] == []


def test_link_below_root_still_fails(tmp_path):
    root = tmp_path / "sealed"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (root / "plain_ece.json").symlink_to(outside)
    with pytest.raises(GuardError, match="symlink"):
        hash_sealed_files(root, "c", "d", ["plain_ece.json"])
