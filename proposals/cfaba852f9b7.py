import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Low-energy queries retain the classifier energy score, while queries whose energy is "
    "already atypical are reranked by train-calibrated pooled-space neighborhood support."
)
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0169", "m_0264"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    e_train = np.asarray(ctx.h.energy(tr.logits), dtype=float).reshape(-1)
    e_eval = np.asarray(ctx.h.energy(ev.logits), dtype=float).reshape(-1)
    e_train = np.nan_to_num(e_train, nan=0.0, posinf=1e6, neginf=-1e6)
    e_eval = np.nan_to_num(e_eval, nan=0.0, posinf=1e6, neginf=-1e6)

    cutoff = float(np.quantile(e_train, 0.70))
    upper = float(np.quantile(e_train, 0.95))
    span = upper - cutoff
    if not np.isfinite(span) or span <= 1e-8:
        span = float(np.std(e_train))
    if not np.isfinite(span) or span <= 1e-8:
        span = 1.0

    n_train = len(tr.labels)
    k_eval = max(1, min(10, n_train))
    k_train = max(1, min(k_eval + 1, n_train))

    d_train = np.asarray(
        ctx.h.knn(tr.pooled, tr.pooled, k=k_train, l2=True), dtype=float
    ).reshape(-1)
    d_eval = np.asarray(
        ctx.h.knn(tr.pooled, ev.pooled, k=k_eval, l2=True), dtype=float
    ).reshape(-1)

    d_train = np.nan_to_num(d_train, nan=0.0, posinf=1e6, neginf=0.0)
    d_eval = np.nan_to_num(d_eval, nan=0.0, posinf=1e6, neginf=0.0)

    reference = np.sort(d_train)
    support_tail = np.searchsorted(reference, d_eval, side="right") / float(len(reference))
    reranked = cutoff + span * support_tail

    return np.where(e_eval <= cutoff, e_eval, reranked)
