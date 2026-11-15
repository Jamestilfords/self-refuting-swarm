import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = (
    "This score models the class-conditional shape of EDL Dirichlet concentrations, "
    "including both relative evidence allocation and total evidence precision. OOS inputs can have atypical concentration profiles even when their energy or vacuity resembles an in-scope example."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    k = a_tr.shape[1]
    y = np.asarray(tr.labels, dtype=int)
    valid = (y >= 0) & (y < k)
    y = y[valid]
    a_tr = a_tr[valid]

    def evidence_features(alpha):
        total = np.sum(alpha, axis=1, keepdims=True)
        proportions = alpha / np.maximum(total, 1e-8)
        clr = np.log(np.maximum(proportions, 1e-8))
        clr = clr - np.mean(clr, axis=1, keepdims=True)
        precision = np.log1p(total)
        return np.concatenate([clr, precision], axis=1)

    z_tr = evidence_features(a_tr)
    z_ev = evidence_features(a_ev)
    d = z_tr.shape[1]

    global_mean = np.mean(z_tr, axis=0)
    global_var = np.var(z_tr, axis=0) + 1e-4
    counts = np.bincount(y, minlength=k).astype(float)

    sums = np.zeros((k, d), dtype=float)
    np.add.at(sums, y, z_tr)
    means = sums / np.maximum(counts[:, None], 1.0)

    centered = z_tr - means[y]
    sq_sums = np.zeros((k, d), dtype=float)
    np.add.at(sq_sums, y, centered * centered)
    vars_ = sq_sums / np.maximum(counts[:, None], 1.0)

    shrink = counts[:, None] / (counts[:, None] + 8.0)
    means = shrink * means + (1.0 - shrink) * global_mean
    vars_ = shrink * vars_ + (1.0 - shrink) * global_var + 1e-4

    pred = np.argmax(ev.logits, axis=1)
    delta = z_ev - means[pred]
    score = np.mean((delta * delta) / vars_[pred], axis=1)
    return np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=0.0)
