"""Exercise the full CLI on synthetic caches, all four cells, and real critic gates."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from skeptic.rules.critic_bridge import ensure_er1_path
ensure_er1_path()
from skeptic.critic.config import CriticConfig
from skeptic.rules.replay import main
from skeptic.score_artifacts import CELLS, auroc
from skeptic.store import GuardError, write_decision_hash
from tests.fixtures.critic.synth import make_cell, strong


def write_score_manifest(root):
    files = sorted((root / "open/scores_isolated").glob("*/*.npz"))
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  runs/scores_isolated/{path.parent.name}/{path.name}\n"
             for path in files]
    manifest = root / "runs/score_manifest.sha256"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text("".join(lines))


@pytest.fixture
def experiment(tmp_path):
    synth = make_cell(n_id=60, n_oos=20)
    names = {name: f"{i + 1:012x}" for i, name in enumerate(synth.baselines.scores)}
    good, guard = "a" * 12, "b" * 12
    rows = []
    for cell, group in CELLS.items():
        folder = tmp_path / "open/scores_isolated" / group
        folder.mkdir(parents=True)
        for name, scoreset in {**synth.baselines.scores, "strong": strong(synth)}.items():
            sha = names.get(name, good)
            values = np.array([np.r_[inside, outside] for inside, outside in zip(scoreset.val_id, scoreset.val_oos)])
            np.savez_compressed(folder / f"{sha}.npz", scores=values, n_in=60,
                                scorer="isolated-e3-v1", proposal_sha=sha)
            rows.append({"sha": sha, "cell": cell, "arm": scoreset.arm, "status": "ok",
                         "stream": "templated" if name in {"strong", "msp_ce"} else "baseline",
                         "val": {"auroc": [auroc(s[:60], s[60:]) for s in values],
                                 "ece": None, "ece_plain": [.02] * 5}})
        rows.append({"sha": guard, "cell": cell, "arm": "ce", "status": "io", "stream": "llm", "val": None})
    config = {"orderings": 2, "ordering_seed": 8, "initial": {c: names["energy_ce"] for c in CELLS},
              "r1_threshold": .01, "r1_split_seed": 7,
              "r1_noise": {"distribution": "laplace", "sigma": .0025, "seed": 0,
                           "algorithm": "numpy-pcg64-seedsequence-v1"},
              "ignore_parents": ["03e0017097f5", "2d5e3b3523bd", "cddf9272a0a9"],
              "critic": asdict(CriticConfig(bootstrap_B=40))}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "baselines.json").write_text(json.dumps([{"name": n.replace("_ce", "@ce").replace("_edl", "@edl"), "sha": sha} for n, sha in names.items()]))
    args = ["--config", str(tmp_path / "config.json"), "--ledger", str(tmp_path / "ledger.jsonl"),
            "--open-root", str(tmp_path / "open"), "--source", "isolated", "--baselines", str(tmp_path / "baselines.json"),
            "--output", str(tmp_path)]
    return tmp_path, args


def test_plan_run_and_freeze_are_separate(experiment, capsys):
    root, args = experiment
    assert main(["plan", *args]) == 0
    assert json.loads(capsys.readouterr().out)["decision_files"] == 6
    assert not (root / "runs").exists()
    with pytest.raises(GuardError, match="Preregistration"):
        main(["run", *args])
    (root / "PREREGISTRATION.md").write_text("Status: FROZEN\nSynthetic test fixture only.\n")
    write_score_manifest(root)
    assert main(["run", *args]) == 0
    paths = list((root / "runs/decisions").glob("R*/ord_*.jsonl"))
    assert len(paths) == 6
    before = {p: p.read_bytes() for p in paths}
    assert all(len(p.read_text().splitlines()) == 4 for p in paths)
    # A registered baseline tagged "templated" in the source ledger is still excluded.
    assert all(json.loads(line)["sha"] == "a" * 12
               for p in paths for line in p.read_text().splitlines())
    r2 = [json.loads(line) for p in paths if p.parent.name == "R2" for line in p.read_text().splitlines()]
    assert {r["verdict"] for r in r2} == {"UNDERPOWERED"}
    manifest = json.loads((root / "runs/decision_manifest.json").read_text())
    assert manifest["replay"]["source"] == "isolated" and len(manifest["replay"]["score_sha256"]) == 40
    assert main(["run", *args]) == 0
    assert before == {p: p.read_bytes() for p in paths}
    write_decision_hash(root)
    with pytest.raises(GuardError, match="frozen"):
        main(["run", *args])


def test_configuration_and_score_mismatch_fail_before_writing(experiment):
    root, args = experiment
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    del config["critic"]["fdr_alpha"]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Every critic setting"):
        main(["plan", *args])
    config["critic"]["fdr_alpha"] = .1
    config["r1_noise"]["sigma"] = 0
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="R1 needs a registered finite Laplace"):
        main(["plan", *args])
    config["r1_noise"]["sigma"] = .0025
    config_path.write_text(json.dumps(config))
    (root / "PREREGISTRATION.md").write_text("Status: FROZEN\n")
    write_score_manifest(root)
    ledger = root / "ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[0]["val"]["auroc"] = [0.] * 5
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="AUROC and score artifact differ"):
        main(["run", *args])
    assert not (root / "runs" / "decisions").exists()


def test_registered_config_plans_200_orderings(capsys):
    root = Path(__file__).resolve().parents[1]
    args = ["--config", str(root / "runs/replay-config.json"),
            "--ledger", str(root / "runs/ledger.jsonl"),
            "--baselines", str(root / "proposals/baselines.json"),
            "--open-root", str(root / "runs"), "--source", "legacy", "--output", str(root)]
    assert main(["plan", *args]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["orderings"] == 200 and summary["decision_files"] == 600
    assert all(count == 487 for count in summary["proposals_per_cell"].values())


def test_changed_score_file_cannot_pass_registered_manifest(experiment):
    root, args = experiment
    (root / "PREREGISTRATION.md").write_text("Status: FROZEN\n")
    write_score_manifest(root)
    path = next((root / "open/scores_isolated").glob("*/*.npz"))
    with np.load(path) as cache:
        values = cache["scores"]
        n_in = int(cache["n_in"])
        scorer = str(cache["scorer"])
        proposal_sha = str(cache["proposal_sha"])
    np.savez_compressed(path, scores=values, n_in=n_in, scorer=scorer,
                        proposal_sha=proposal_sha, runtime_s=1.0)
    with pytest.raises(ValueError, match="Score artifact differs from registered manifest"):
        main(["run", *args])
    assert not (root / "runs/decisions").exists()
