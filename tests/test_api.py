"""L1 done-when: the slide-8 example runs, and each validator rule has a failing test case."""
import textwrap

import numpy as np
import pytest

from skeptic import api

HEADER = textwrap.dedent('''
    FAMILY = "fusion"
    ARM = "ce"
    RATIONALE = "test"
    SOURCE = ""
    PARENTS = []
    MEMORY_USED = []
''')
SLIDE8 = HEADER + textwrap.dedent('''
    def score(ctx):
        tr, ev = ctx.train, ctx.eval
        g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
        d = g.min_mahalanobis_sq(ev.pooled)
        return ctx.h.energy(ev.logits) + 0.1 * d
''')


def body(code: str, header: str = HEADER) -> str:
    return header + "\ndef score(ctx):\n" + textwrap.indent(textwrap.dedent(code).strip(), "    ") + "\n"


@pytest.fixture(scope="module")
def ctx():
    rng = np.random.default_rng(0)
    k, n_tr, n_ev = 5, 400, 120

    def split(n, shift=0.0):
        y = rng.integers(0, k, n)
        mu_f, mu_p = np.eye(k, 16) * 4, np.eye(k, 32) * 4
        feats = mu_f[y] + rng.normal(size=(n, 16)) + shift
        pooled = mu_p[y] + rng.normal(size=(n, 32)) + shift
        logits = np.eye(k)[y] * 6 + rng.normal(size=(n, k))
        return {"logits": logits, "feats": feats, "pooled": pooled, "labels": y}

    ev = split(n_ev, shift=0.5)
    ev["labels"] = np.full(n_ev, -1)          # must be dropped by make_ctx
    return api.make_ctx(split(n_tr), ev)


def write(tmp_path, text):
    p = tmp_path / f"{api.proposal_sha(text)}.py"
    p.write_text(text, encoding="utf-8")
    return p


def load(tmp_path, text):
    return api.load_proposal(write(tmp_path, text))


def reason(fn):
    with pytest.raises(api.Malformed) as e:
        fn()
    return e.value.reason


# ---------------------------------------------------------------- the example
def test_slide8_example_runs(tmp_path, ctx):
    p = load(tmp_path, SLIDE8)
    out = api.validate(p, ctx)
    assert out["scores"].shape == (len(ctx.eval),) and out["probs"] is None
    assert p.family == "fusion" and p.arm == "ce" and len(p.sha) == 12


def test_template_file_is_valid():
    api.static_check(open("proposals/_template.py", encoding="utf-8").read())


def test_ctx_hides_labels_and_is_read_only(ctx):
    assert ctx.eval.labels is None and not hasattr(ctx.eval, "split")
    with pytest.raises(ValueError):
        ctx.train.logits[0, 0] = 1.0


# ---------------------------------------------------------------- one failing case per rule
def test_wrong_length(tmp_path, ctx):
    p = load(tmp_path, body("return ctx.h.energy(ctx.eval.logits)[:-1]"))
    assert reason(lambda: api.validate(p, ctx)) == "length"


def test_nan(tmp_path, ctx):
    p = load(tmp_path, body("s = ctx.h.energy(ctx.eval.logits)\nreturn s * float('nan')"))
    assert reason(lambda: api.validate(p, ctx)) == "nan"


def test_constant(tmp_path, ctx):
    p = load(tmp_path, body("import numpy as np\nreturn np.zeros(len(ctx.eval))"))
    assert reason(lambda: api.validate(p, ctx)) == "constant"


@pytest.mark.parametrize("line", ["import os", "import torch", "from pathlib import Path", "import numpy.ctypeslib",
                                  "from sklearn.datasets import fetch_20newsgroups"])
def test_import_outside_allowlist(tmp_path, line):
    text = line + "\n" + SLIDE8
    assert reason(lambda: load(tmp_path, text)) in ("import", "io")


@pytest.mark.parametrize("code", [
    "open('/sealed/test.npz')\nreturn ctx.h.energy(ctx.eval.logits)",
    "import numpy as np\nnp.load('/sealed/cache.npz')\nreturn ctx.h.energy(ctx.eval.logits)",
    "x = ctx.h.__class__\nreturn ctx.h.energy(ctx.eval.logits)",
    "return eval('1')",
])
def test_file_or_network_access(tmp_path, code):
    assert reason(lambda: load(tmp_path, body(code))) == "io"


def test_timeout(tmp_path, ctx, monkeypatch):
    monkeypatch.setattr(api, "MAX_SECONDS", 0.0)
    p = load(tmp_path, SLIDE8)
    assert reason(lambda: api.validate(p, ctx)) == "timeout"


def test_subset_dependent(tmp_path, ctx):
    p = load(tmp_path, body("s = ctx.h.energy(ctx.eval.logits)\nreturn s - s.mean()"))
    assert reason(lambda: api.validate(p, ctx)) == "subset_dependent"


def test_runtime_error(tmp_path, ctx):
    p = load(tmp_path, body("return ctx.eval.labels + 1"))      # labels are None
    assert reason(lambda: api.validate(p, ctx)) == "runtime_error"


@pytest.mark.parametrize("old,new", [('FAMILY = "fusion"', 'FAMILY = "magic"'), ('ARM = "ce"', 'ARM = "both"'),
                                     ('RATIONALE = "test"', 'RATIONALE = ""'), ('SOURCE = ""', 'SOURCE = "a paper"'),
                                     ('PARENTS = []', 'PARENTS = "energy"'), ('MEMORY_USED = []\n', '')])
def test_bad_header(tmp_path, old, new):
    assert reason(lambda: load(tmp_path, SLIDE8.replace(old, new))) == "header"


def test_wrong_file_name(tmp_path):
    p = tmp_path / "my_idea.py"
    p.write_text(SLIDE8, encoding="utf-8")
    assert reason(lambda: api.load_proposal(p)) == "name"


def test_bad_probs(tmp_path, ctx):
    p = load(tmp_path, body("s = ctx.h.energy(ctx.eval.logits)\nreturn s, ctx.h.softmax(ctx.eval.logits) * 2"))
    assert reason(lambda: api.validate(p, ctx)) == "probs"


# ---------------------------------------------------------------- helpers
def test_logit_scores(ctx):
    h, z = ctx.h, ctx.eval.logits
    assert np.allclose(h.energy(z), -np.log(np.exp(z).sum(1)))
    assert np.allclose(h.msp(z), 1 - h.softmax(z).max(1))
    # ER-1's identity: vacuity is a monotone function of clipped energy, so they rank identically
    assert (np.argsort(h.vacuity(z)) == np.argsort(h.energy_clip(z))).all()


def test_densities_and_distances(ctx):
    h, tr, ev = ctx.h, ctx.train, ctx.eval
    g = h.gda(tr.pooled, tr.labels, pca=8)
    assert g.class_mahalanobis_sq(ev.pooled).shape == (len(ev), 5)
    assert (g.min_mahalanobis_sq(ev.pooled) >= 0).all()
    assert np.allclose(g.neg_log_density(ev.pooled), -g.log_density(ev.pooled))
    for s in (h.relative_mahalanobis(tr.feats, tr.labels, ev.feats), h.mahalanobis_pp(tr.feats, tr.labels, ev.feats),
              h.partial_mahalanobis(tr.feats, tr.labels, ev.feats, top_k=8)["bottom"], h.knn(tr.feats, ev.feats, k=10)):
        assert s.shape == (len(ev),) and np.isfinite(s).all()
    assert h.pca(tr.pooled, 8).transform(ev.pooled).shape == (len(ev), 8)


@pytest.mark.parametrize("name", ["minmax", "qsigmoid", "ecdf"])
def test_normalizers_and_couplings(ctx, name):
    h, tr, ev = ctx.h, ctx.train, ctx.eval
    g = h.gda(tr.feats, tr.labels)
    p_hat = h.normalizer(name, g.log_density(tr.feats))(g.log_density(ev.feats))
    assert ((p_hat >= 0) & (p_hat <= 1)).all()
    s_add, p_add = h.couple_add_log(ev.logits, p_hat, lam=5.0)
    s_mul, p_mul = h.couple_mul(ev.logits, p_hat)
    assert np.allclose(p_add, h.softmax(np.clip(ev.logits, -15, 15)))     # add_log never changes predictions
    assert s_add.shape == s_mul.shape == (len(ev),) and np.allclose(p_mul.sum(1), 1)


def test_coupling_proposal_switches_on_probs(tmp_path, ctx):
    code = """
        tr, ev = ctx.train, ctx.eval
        g = ctx.h.gda(tr.feats, tr.labels)
        p_hat = ctx.h.normalizer("ecdf", g.log_density(tr.feats))(g.log_density(ev.feats))
        return ctx.h.couple_add_log(ev.logits, p_hat, 5.0)
    """
    out = api.validate(load(tmp_path, body(code)), ctx)
    assert out["probs"].shape == (len(ctx.eval), 5)


def test_wrappers_match_er1(ctx):
    """Chunked and re-stated helpers give ER-1's own numbers."""
    _, _, variants = api._er1()
    h, tr, ev = ctx.h, ctx.train, ctx.eval
    assert np.allclose(h.relative_mahalanobis(tr.feats, tr.labels, ev.feats, pca=8),
                       variants.relative_mahalanobis(tr.feats, tr.labels, ev.feats, pca_dim=8))
    assert np.allclose(h.mahalanobis_pp(tr.feats, tr.labels, ev.feats),
                       variants.mahalanobis_pp(tr.feats, tr.labels, ev.feats))
    ref = variants.partial_mahalanobis(tr.pooled, tr.labels, ev.pooled, top_k=8)
    p = h.partial(tr.pooled, tr.labels, top_k=8)
    assert np.allclose(p.top(ev.pooled), ref["top"]) and np.allclose(p.bottom(ev.pooled), ref["bottom"])


def test_chunked_queries_match_unchunked(ctx, monkeypatch):
    g = ctx.h.gda(ctx.train.pooled, ctx.train.labels)
    whole = g.log_density(ctx.eval.pooled)
    monkeypatch.setattr(api, "_chunks", lambda n, width, budget=0: [slice(i, i + 7) for i in range(0, n, 7)])
    assert np.allclose(g.log_density(ctx.eval.pooled), whole)
    assert np.allclose(g.min_mahalanobis_sq(ctx.eval.pooled), g._m.class_sq_mahalanobis(ctx.eval.pooled).min(1))
