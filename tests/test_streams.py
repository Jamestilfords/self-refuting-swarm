"""L2 and L3: the canaries and the templated stream are deterministic, well formed and runnable."""
import json

import numpy as np
import pytest

from skeptic import api
from skeptic.streams import canaries, templated


@pytest.fixture(scope="module")
def ctx():
    """Wide enough (160-d and 200-d) for PCA 128 and the partial-Mahalanobis remainder to exist."""
    rng = np.random.default_rng(1)
    k = 6

    def split(n, shift=0.0):
        y = rng.integers(0, k, n)
        return {"logits": np.eye(k)[y] * 6 + rng.normal(size=(n, k)), "labels": y,
                "feats": np.eye(k, 160)[y] * 4 + rng.normal(size=(n, 160)) + shift,
                "pooled": np.eye(k, 200)[y] * 4 + rng.normal(size=(n, 200)) + shift}

    # Eval mixes in-scope-like and shifted items, as validation does. If every item were shifted,
    # minmax would clip p_hat to 0 everywhere and the mul coupling would be constant (the blog's uniform case).
    inside, outside = split(100), split(50, shift=0.4)
    ev = {key: np.concatenate([inside[key], outside[key]]) for key in inside}
    return api.make_ctx(split(900), ev)


# ---------------------------------------------------------------- L2 canaries
def test_canaries_written_with_expected_verdicts(tmp_path, ctx):
    rows = canaries.write_canaries(tmp_path)
    assert [r["expected"] for r in rows] == ["INERT", "SCORE-MATCHED", "NO EFFECT", "BUG"]
    assert json.loads((tmp_path / "canaries.json").read_text()) == rows
    by_name = {r["name"]: r for r in rows}
    for name in ("inert_rescale", "msp_renamed", "gaussian_noise"):
        out = api.validate(api.load_proposal(tmp_path / f"{by_name[name]['sha']}.py"), ctx)
        assert out["scores"].shape == (len(ctx.eval),)


def test_canary_behaviour(tmp_path, ctx):
    rows = {r["name"]: r for r in canaries.write_canaries(tmp_path)}
    run = lambda n: api.validate(api.load_proposal(tmp_path / f"{rows[n]['sha']}.py"), ctx)["scores"]
    energy = ctx.h.energy(ctx.eval.logits)
    assert (np.argsort(run("inert_rescale")) == np.argsort(energy)).all()          # same ranking as the incumbent
    assert np.allclose(run("msp_renamed"), ctx.h.msp(ctx.eval.logits))             # identical to a baseline
    noise = run("gaussian_noise")
    assert abs(np.corrcoef(noise, energy)[0, 1]) < 0.3 and abs(noise.mean()) < 0.4  # unrelated to the incumbent


def test_leak_canary_is_stopped(tmp_path):
    row = [r for r in canaries.write_canaries(tmp_path) if r["name"] == "test_leak"][0]
    assert row["validator_status"] == "io"
    with pytest.raises(api.Malformed) as e:
        api.load_proposal(tmp_path / f"{row['sha']}.py")
    assert e.value.reason == "io"


# ---------------------------------------------------------------- L3 templated stream
def test_grid_size_and_uniqueness():
    g = templated.grid()
    assert len(g) == 896 == len(set(g))
    assert len({templated.render_config(c) for c in g}) == 896          # every configuration is a distinct file


def test_every_grid_file_passes_static_checks():
    for c in templated.grid():
        api.static_check(templated.render_config(c))


def test_build_is_deterministic(tmp_path):
    a = templated.build(tmp_path / "a")
    b = templated.build(tmp_path / "b")
    assert a == b
    assert sum("draw" in p["roles"] for p in a["proposals"]) == 300
    assert [s["n"] for s in a["scorecard"]] == list(range(1, 9))
    shas = {p["sha"] for p in a["proposals"]}
    assert all(s["candidate"] in shas and s["incumbent"] in shas for s in a["scorecard"])
    files = {p.stem for p in (tmp_path / "a").glob("*.py")}
    assert files == shas
    for sha in shas:                                                     # file name is the content hash
        api.load_proposal(tmp_path / "a" / f"{sha}.py")


def test_different_seed_gives_a_different_draw(tmp_path):
    a, b = templated.build(tmp_path / "a", seed=0), templated.build(tmp_path / "b", seed=1)
    draw = lambda m: {p["sha"] for p in m["proposals"] if "draw" in p["roles"]}
    assert draw(a) != draw(b)


def _cases():
    seen, out = set(), []
    for c in templated.grid():                       # one of each feature x pca x variant x coupling
        key = (c.feature, c.pca, c.variant, c.coupling)
        if key not in seen and (c.coupling != "add_log" or c.lam == 5):
            seen.add(key)
            out.append(c)
    return out + [item for _, _, cand, inc, _, _ in templated.SCORECARD for item in (cand, inc)]


@pytest.mark.parametrize("item", _cases(), ids=str)
def test_templates_run_and_validate(tmp_path, ctx, item):
    text = templated._text(item)
    path = tmp_path / f"{api.proposal_sha(text)}.py"
    path.write_text(text, encoding="utf-8")
    out = api.validate(api.load_proposal(path), ctx)
    assert np.isfinite(out["scores"]).all()
    if isinstance(item, templated.Config) and item.coupling != "none":
        assert out["probs"] is not None               # couplings return probabilities, which switches on G5
