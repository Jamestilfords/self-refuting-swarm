"""Score-function API, helpers and validator for Skeptic proposals (work package L1).

A proposal is one Python file that defines six constants and `score(ctx)`:

    ctx.train   logits, feats (256-d bottleneck), pooled (768-d encoder output), labels. Fit everything here.
    ctx.eval    logits, feats, pooled for the items to score. No labels and no split name.
    ctx.h       helpers below (energy, msp, gda, knn, pca, Mahalanobis variants, normalizers, couplings).
    return      one float per eval item, higher = more out-of-scope; or (scores, probs) to switch on gate G5.

The density code is ER-1's own (github.com/mid9t/ER-1-Epistemic-Ranker-, MIT), imported from the pinned
submodule at third_party/er1 so numbers match the paper. Logit scores are re-stated in numpy from
ER-1's uncertainty/scores.py so the sandbox needs only numpy, scipy and scikit-learn.

Isolation note: the checks here reject malformed or careless proposals. They are not a security boundary.
The sandbox (no test arrays mounted, no network) is what enforces the lock.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax as _softmax

FAMILIES = ("logit", "density", "knn", "fusion", "other")
ARMS = ("ce", "edl")
ALLOWED_IMPORTS = ("numpy", "scipy", "sklearn")
BANNED_MODULES = ("scipy.io", "scipy.datasets", "sklearn.datasets", "sklearn.externals",
                  "numpy.ctypeslib", "numpy.f2py", "numpy.distutils", "numpy.lib.npyio", "numpy.testing")
BANNED_NAMES = ("open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals",
                "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit", "help")
BANNED_ATTRS = ("load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt",
                "fromfile", "tofile", "memmap", "DataSource", "dump", "dumps", "loads", "fromregex")
MAX_SECONDS = 60.0     # † per seed
SUBSET_FRAC = 0.10     # † scores must not change when a 10% subset is scored alone
LOGIT_CLAMP, ALPHA_EPS = 15.0, 1e-6   # ER-1 posthoc/scoring.py


class Malformed(Exception):
    """A proposal broke a rule on slide 8. `reason` is a short code the ledger stores as the status."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


# ------------------------------------------------------------------ ER-1 import
def _er1():
    """Import ER-1's numpy-only density modules from the pinned submodule."""
    src = os.environ.get("SKEPTIC_ER1_SRC") or str(Path(__file__).resolve().parents[1] / "third_party" / "er1" / "src")
    if not Path(src, "bert_daedl").is_dir():
        raise RuntimeError(f"ER-1 not found at {src}. Run: git submodule update --init")
    if src not in sys.path:
        sys.path.insert(0, src)
    # density/gda.py imports torch for one function we never call (scale_alpha). If torch is missing,
    # a stub is present only while ER-1 imports, then removed so numpy/scipy never see a fake torch.
    stub = None
    try:
        import torch  # noqa: F401
    except ImportError:
        stub = sys.modules["torch"] = types.ModuleType("torch")
    try:
        return (importlib.import_module("bert_daedl.config"),
                importlib.import_module("bert_daedl.density.gda"),
                importlib.import_module("bert_daedl.posthoc.gda_variants"))
    finally:
        if stub is not None and sys.modules.get("torch") is stub:
            del sys.modules["torch"]


# ------------------------------------------------------------------ context
def _ro(a, dtype=np.float64):
    out = np.array(a, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class Split:
    logits: np.ndarray
    feats: np.ndarray
    pooled: np.ndarray
    labels: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.logits)


@dataclass(frozen=True)
class Ctx:
    train: Split
    eval: Split
    h: "Helpers"


def make_ctx(train: dict, eval: dict) -> Ctx:
    """Build a context from two dicts of arrays. Eval labels and any other keys are dropped here,
    so a score function cannot tell validation from test or in-scope from OOS."""
    tr = Split(_ro(train["logits"]), _ro(train["feats"]), _ro(train["pooled"]), _ro(train["labels"], np.int64))
    ev = Split(_ro(eval["logits"]), _ro(eval["feats"]), _ro(eval["pooled"]), None)
    if not (len(tr.feats) == len(tr.pooled) == len(tr.labels) == len(tr)) or not (len(ev.feats) == len(ev.pooled) == len(ev)):
        raise ValueError("arrays in a split must have the same length")
    return Ctx(tr, ev, Helpers())


def subset_ctx(ctx: Ctx, idx) -> Ctx:
    ev = ctx.eval
    return Ctx(ctx.train, Split(_ro(ev.logits[idx]), _ro(ev.feats[idx]), _ro(ev.pooled[idx]), None), ctx.h)


# ------------------------------------------------------------------ helpers
class GDA:
    """Fitted Gaussian discriminant density (ER-1 GaussianDensityModel)."""

    def __init__(self, model):
        self._m = model

    def class_mahalanobis_sq(self, x) -> np.ndarray:      # (N, C)
        return self._m.class_sq_mahalanobis(x)

    def min_mahalanobis_sq(self, x) -> np.ndarray:        # (N,) higher = more OOS
        return self._m.class_sq_mahalanobis(x).min(axis=1)

    def log_density(self, x) -> np.ndarray:               # (N,) higher = more like training data
        return self._m.marginal_log_prob(x)

    def neg_log_density(self, x) -> np.ndarray:           # (N,) higher = more OOS
        return -self._m.marginal_log_prob(x)


class PCA:
    """Centre on train, project on the top `dim` principal components (no whitening)."""

    def __init__(self, x, dim: int):
        X = np.asarray(x, np.float64)
        self.mean_ = X.mean(0)
        w, V = np.linalg.eigh(np.cov(X - self.mean_, rowvar=False))
        self.components_ = V[:, np.argsort(w)[::-1][: min(dim, X.shape[1])]]

    def transform(self, x) -> np.ndarray:
        return (np.asarray(x, np.float64) - self.mean_) @ self.components_


class Helpers:
    """Every score follows one convention: higher = more out-of-scope."""

    # ---- logit scores (ER-1 uncertainty/scores.py)
    @staticmethod
    def softmax(logits) -> np.ndarray:
        return _softmax(np.asarray(logits, np.float64), axis=-1)

    @staticmethod
    def energy(logits) -> np.ndarray:
        return -logsumexp(np.asarray(logits, np.float64), axis=-1)

    @staticmethod
    def energy_clip(logits, clamp: float = LOGIT_CLAMP) -> np.ndarray:
        return -logsumexp(np.clip(np.asarray(logits, np.float64), -clamp, clamp), axis=-1)

    @staticmethod
    def msp(logits) -> np.ndarray:
        return 1.0 - Helpers.softmax(logits).max(axis=-1)

    @staticmethod
    def entropy(logits) -> np.ndarray:
        p = Helpers.softmax(logits)
        return -(p * np.log(p + 1e-12)).sum(axis=-1)

    @staticmethod
    def alpha(logits, clamp: float = LOGIT_CLAMP, eps: float = ALPHA_EPS) -> np.ndarray:
        return np.exp(np.clip(np.asarray(logits, np.float64), -clamp, clamp)) + eps

    @staticmethod
    def vacuity(logits) -> np.ndarray:
        a = Helpers.alpha(logits)
        return a.shape[-1] / a.sum(axis=-1)

    # ---- densities and distances (ER-1 density/gda.py, posthoc/gda_variants.py)
    @staticmethod
    def gda(x, labels, cov: str = "tied", pca: int | None = None, whiten: bool = True) -> GDA:
        cfg_mod, gda_mod, _ = _er1()
        cfg = cfg_mod.GDAConfig(cov_mode=cov, pca_dim=pca, whiten=whiten)
        return GDA(gda_mod.GaussianDensityModel(cfg).fit(np.asarray(x, np.float64), np.asarray(labels)))

    @staticmethod
    def relative_mahalanobis(train_x, labels, query_x, pca: int | None = None) -> np.ndarray:
        return _er1()[2].relative_mahalanobis(train_x, labels, query_x, pca_dim=pca)

    @staticmethod
    def mahalanobis_pp(train_x, labels, query_x, pca: int | None = None) -> np.ndarray:
        return _er1()[2].mahalanobis_pp(train_x, labels, query_x, pca_dim=pca)

    @staticmethod
    def partial_mahalanobis(train_x, labels, query_x, top_k: int = 128) -> dict:
        return _er1()[2].partial_mahalanobis(train_x, labels, query_x, top_k=top_k)   # {"top", "bottom"}

    @staticmethod
    def knn(train_x, query_x, k: int = 10, l2: bool = True) -> np.ndarray:
        """Distance to the k-th nearest training item. Not in ER-1; standard kNN OOD score."""
        from sklearn.neighbors import NearestNeighbors
        norm = (lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)) if l2 else (lambda a: a)
        A, Q = norm(np.asarray(train_x, np.float64)), norm(np.asarray(query_x, np.float64))
        return NearestNeighbors(n_neighbors=k).fit(A).kneighbors(Q)[0][:, -1]

    @staticmethod
    def pca(train_x, dim: int) -> PCA:
        return PCA(train_x, dim)

    # ---- normalizers: map a log-density to p_hat in [0, 1], 1 = looks like training data
    @staticmethod
    def normalizer(name: str, train_log_density):
        norms = _er1()[1].NORMALIZERS
        if name not in norms:
            raise ValueError(f"normalizer must be one of {sorted(norms)}")
        return norms[name]().fit(np.asarray(train_log_density, np.float64))

    # ---- couplings: combine evidence with a density. Both return (scores, probs), which switches on G5.
    @staticmethod
    def couple_mul(logits, p_hat):
        """DAEDL: alpha = exp(p_hat * z). Temperature scaling, so it changes the predicted probabilities."""
        z = np.clip(np.asarray(logits, np.float64), -LOGIT_CLAMP, LOGIT_CLAMP) * np.asarray(p_hat, np.float64)[:, None]
        a = np.exp(z)
        return a.shape[-1] / a.sum(-1), a / a.sum(-1, keepdims=True)

    @staticmethod
    def couple_add_log(logits, p_hat, lam: float):
        """Additive: log vacuity = log K + energy_clip + lam * (1 - p_hat). Probabilities are unchanged."""
        z = np.clip(np.asarray(logits, np.float64), -LOGIT_CLAMP, LOGIT_CLAMP)
        score = np.log(z.shape[-1]) - logsumexp(z, axis=-1) + lam * (1.0 - np.asarray(p_hat, np.float64))
        return score, _softmax(z, axis=-1)


# ------------------------------------------------------------------ proposal files
@dataclass(frozen=True)
class Proposal:
    sha: str
    path: str
    family: str
    arm: str
    rationale: str
    source: str
    parents: tuple
    memory_used: tuple
    score: object
    text: str


def proposal_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _str_list(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


_HEADER = {
    "FAMILY": lambda v: v in FAMILIES,
    "ARM": lambda v: v in ARMS,
    "RATIONALE": lambda v: isinstance(v, str) and v.strip() != "",
    "SOURCE": lambda v: isinstance(v, str) and (v == "" or v.startswith(("http://", "https://"))),
    "PARENTS": _str_list,
    "MEMORY_USED": _str_list,
}


def static_check(text: str) -> dict:
    """Parse without running. Returns the header constants. Raises Malformed."""
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise Malformed("syntax", str(e))
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise Malformed("import", "relative import")
            mods = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
        for m in mods:
            if m.split(".")[0] not in ALLOWED_IMPORTS:
                raise Malformed("import", f"{m} is outside numpy, scipy, sklearn")
            if any(m == b or m.startswith(b + ".") for b in BANNED_MODULES):
                raise Malformed("io", f"{m} can read files or the network")
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise Malformed("io", f"use of {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr in BANNED_ATTRS or node.attr.startswith("__")):
            raise Malformed("io", f"attribute .{node.attr}")
    header = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in _HEADER:
                try:
                    header[name] = ast.literal_eval(node.value)
                except ValueError:
                    raise Malformed("header", f"{name} must be a literal")
    for name, ok in _HEADER.items():
        if name not in header:
            raise Malformed("header", f"missing {name}")
        if not ok(header[name]):
            raise Malformed("header", f"bad value for {name}: {header[name]!r}")
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "score"]
    if len(fns) != 1 or [a.arg for a in fns[0].args.args] != ["ctx"]:
        raise Malformed("header", "need exactly one top-level def score(ctx)")
    return header


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED_IMPORTS or any(name == b or name.startswith(b + ".") for b in BANNED_MODULES):
        raise ImportError(f"import of {name} is not allowed in a proposal")
    return importlib.__import__(name, globals, locals, fromlist, level)


_SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len", "list", "map",
    "max", "min", "pow", "print", "range", "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "zip", "ValueError", "TypeError", "ZeroDivisionError", "ArithmeticError", "Exception")}
_SAFE_BUILTINS["__import__"] = _guarded_import


def load_proposal(path, check_name: bool = True) -> Proposal:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    sha = proposal_sha(text)
    if check_name and path.stem != sha:
        raise Malformed("name", f"file must be named {sha}.py (first 12 hex of its SHA-256)")
    h = static_check(text)
    ns = {"__builtins__": dict(_SAFE_BUILTINS), "__name__": f"proposal_{sha}"}
    try:
        exec(compile(text, str(path), "exec"), ns)  # noqa: S102 - statically checked above; runs in the sandbox
    except Exception as e:
        raise Malformed("import_error", f"{type(e).__name__}: {e}")
    return Proposal(sha, str(path), h["FAMILY"], h["ARM"], h["RATIONALE"], h["SOURCE"],
                    tuple(h["PARENTS"]), tuple(h["MEMORY_USED"]), ns["score"], text)


def run_proposal(p: Proposal, ctx: Ctx):
    """Run score(ctx) and check its output. Returns (scores, probs or None, seconds). Raises Malformed."""
    t0 = time.perf_counter()
    try:
        out = p.score(ctx)
    except Exception as e:
        raise Malformed("runtime_error", f"{type(e).__name__}: {e}")
    seconds = time.perf_counter() - t0
    if seconds > MAX_SECONDS:
        raise Malformed("timeout", f"{seconds:.1f}s > {MAX_SECONDS:.0f}s")
    probs = None
    if isinstance(out, tuple):
        if len(out) != 2:
            raise Malformed("length", "return scores or (scores, probs)")
        out, probs = out
    try:
        scores = np.asarray(out, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise Malformed("length", f"scores are not numeric: {e}")
    n = len(ctx.eval)
    if scores.shape != (n,):
        raise Malformed("length", f"expected shape ({n},), got {scores.shape}")
    if not np.isfinite(scores).all():
        raise Malformed("nan", "scores contain NaN or inf")
    if np.ptp(scores) == 0:
        raise Malformed("constant", "every item gets the same score")
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float64)
        k = ctx.train.logits.shape[1]
        if probs.shape != (n, k) or not np.isfinite(probs).all() or (probs < 0).any() \
                or not np.allclose(probs.sum(1), 1.0, atol=1e-4):
            raise Malformed("probs", f"probs must be ({n}, {k}), non-negative, rows summing to 1")
    return scores, probs, seconds


def validate(p: Proposal, ctx: Ctx, seed: int = 0) -> dict:
    """All dynamic rules of slide 8. Returns scores, probs and runtime. Raises Malformed."""
    scores, probs, seconds = run_proposal(p, ctx)
    n = len(ctx.eval)
    idx = np.sort(np.random.default_rng(seed).choice(n, size=max(2, int(np.ceil(SUBSET_FRAC * n))), replace=False))
    try:
        sub = np.asarray(p.score(subset_ctx(ctx, idx))[0] if probs is not None else p.score(subset_ctx(ctx, idx)), np.float64)
    except Exception as e:
        raise Malformed("subset_dependent", f"fails on a subset: {type(e).__name__}: {e}")
    if sub.shape != (len(idx),) or not np.allclose(sub, scores[idx], rtol=1e-6, atol=1e-8):
        raise Malformed("subset_dependent", "scores change when a 10% subset is scored alone "
                                            "(the function is using statistics of the eval set)")
    return {"scores": scores, "probs": probs, "runtime_s": round(seconds, 3)}


# ------------------------------------------------------------------ CLI
def _main(argv) -> int:
    if len(argv) < 2 or argv[0] not in ("check", "name"):
        print("usage: python -m skeptic.api check FILE...   static checks and file name\n"
              "       python -m skeptic.api name FILE...    rename each file to <sha>.py")
        return 2
    bad = 0
    for f in argv[1:]:
        path = Path(f)
        try:
            if argv[0] == "name":
                text = path.read_text(encoding="utf-8")
                static_check(text)
                new = path.with_name(proposal_sha(text) + ".py")
                path.rename(new)
                print(f"ok   {f} -> {new}")
            else:
                p = load_proposal(path)
                print(f"ok   {p.sha}  family={p.family} arm={p.arm} parents={list(p.parents)}")
        except Malformed as e:
            bad += 1
            print(f"FAIL {f}: {e}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
