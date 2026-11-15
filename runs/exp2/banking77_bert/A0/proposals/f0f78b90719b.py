import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions encode tight constraints that in-scope representations should satisfy. A large fraction of residual energy concentrated in these trailing directions indicates poor compatibility with known intent manifolds."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0049"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)

    def row_energy(x):
        a = np.asarray(x, dtype=float)
        if a.ndim == 1:
            return a
        return np.sum(a * a, axis=1)

    train_top = row_energy(partial.top(tr.pooled))
    train_tail = row_energy(partial.bottom(tr.pooled))
    eval_top = row_energy(partial.top(ev.pooled))
    eval_tail = row_energy(partial.bottom(ev.pooled))

    eps = max(float(np.median(train_top + train_tail)) * 1e-10, 1e-12)

    train_fraction = train_tail / (train_top + train_tail + eps)
    eval_fraction = eval_tail / (eval_top + eval_tail + eps)

    train_log_tail = np.log1p(np.maximum(train_tail, 0.0))
    eval_log_tail = np.log1p(np.maximum(eval_tail, 0.0))

    frac_center = np.median(train_fraction)
    frac_scale = np.median(np.abs(train_fraction - frac_center))
    if not np.isfinite(frac_scale) or frac_scale <= 1e-12:
        frac_scale = np.std(train_fraction) + 1e-12

    tail_center = np.median(train_log_tail)
    tail_scale = np.median(np.abs(train_log_tail - tail_center))
    if not np.isfinite(tail_scale) or tail_scale <= 1e-12:
        tail_scale = np.std(train_log_tail) + 1e-12

    fraction_score = (eval_fraction - frac_center) / frac_scale
    magnitude_score = (eval_log_tail - tail_center) / tail_scale
    return np.nan_to_num(0.8 * fraction_score + 0.2 * magnitude_score,
                         nan=0.0, posinf=50.0, neginf=-50.0)
