import numpy as np

FAMILY = "knn"
ARM = "edl"
RATIONALE = (
    "OOS inputs can have Dirichlet concentration profiles unlike the evidence-shape patterns "
    "seen for any in-scope class, even when their total evidence or energy is similar."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    def profile(alpha):
        log_a = np.log(alpha)
        centered = log_a - log_a.mean(axis=1, keepdims=True)
        log_total = np.log(alpha.sum(axis=1, keepdims=True))
        concentration = np.std(log_a, axis=1, keepdims=True)
        return np.concatenate([centered, log_total, concentration], axis=1)

    z_tr = profile(a_tr)
    z_ev = profile(a_ev)

    med = np.median(z_tr, axis=0, keepdims=True)
    scale = np.median(np.abs(z_tr - med), axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-5)

    z_tr = (z_tr - med) / scale
    z_ev = (z_ev - med) / scale

    base = ctx.h.knn(z_tr, z_ev, k=10, l2=False)
    tie_break = 1e-10 * np.sum(z_ev * np.arange(1, z_ev.shape[1] + 1), axis=1)
    return np.asarray(base, dtype=float) + tie_break
