"""Same inputs, same decision bytes, including across a warm cache and a new process."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from skeptic.critic import CriticConfig, JsonDirStatsCache, LordPlusPlus, MemoryStatsCache, verdict
from tests.fixtures.critic.synth import make_cell, strong

FAST = CriticConfig(bootstrap_B=200)
ROOT = Path(__file__).resolve().parents[2]


def _run(cache=None):
    cell = make_cell()
    return verdict(
        strong(cell),
        cell.incumbent,
        cell.baselines,
        LordPlusPlus(),
        cell.cell,
        config=FAST,
        cache=cache,
    )


def test_repeat_calls_match():
    assert _run().to_json() == _run().to_json()


def test_cache_matches_and_suppresses_the_second_bootstrap(monkeypatch, tmp_path):
    first = _run()
    memory = MemoryStatsCache()
    second = _run(cache=memory)
    folder = JsonDirStatsCache(tmp_path)
    on_disk = _run(cache=folder)
    assert first.to_json() == second.to_json() == on_disk.to_json()

    def boom(*args, **kwargs):
        raise AssertionError("warm cache should not bootstrap")

    monkeypatch.setattr("skeptic.critic.er1.paired_bootstrap", boom)
    assert _run(cache=memory).to_json() == first.to_json()
    assert _run(cache=folder).to_json() == first.to_json()


def test_fresh_process_matches():
    expected = _run().to_json()
    code = """
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / "third_party" / "er1" / "src")]
from skeptic.critic import CriticConfig, LordPlusPlus, verdict
from tests.fixtures.critic.synth import make_cell, strong
cell = make_cell()
config = CriticConfig(bootstrap_B=200)
decision = verdict(strong(cell), cell.incumbent, cell.baselines, LordPlusPlus(), cell.cell, config=config)
sys.stdout.write(decision.to_json())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "third_party" / "er1" / "src")])
    completed = subprocess.run(
        [sys.executable, "-c", code, str(ROOT)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.stdout == expected
