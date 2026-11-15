import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "For confident predictions the score is exactly energy, while uncertain predictions receive "
    "a feature-support residual measuring Mahalanobis distance beyond what in-scope uncertainty predicts."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0100", "m_0115"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_d = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_d = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    train_e = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_e = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_h = np.asarray(ctx.h.entropy(tr.logits), dtype=float)
    eval_h = np.asarray(ctx.h.entropy(ev.logits), dtype=float)

    x = np.column_stack([np.ones(train_h.shape[0]), train_h])
    coef, _, _, _ = np.linalg.lstsq(x, train_d, rcond=None)

    train_resid = train_d - x @ coef
    center = float(np.median(train_resid))
    lo, hi = np.quantile(train_resid, [0.25, 0.75])
    scale = float(max(hi - lo, 1e-6))

    uncertainty_cutoff = float(np.quantile(train_h, 0.70))
    uncertain = (eval_h > uncertainty_cutoff).astype(float)

    eval_expected_d = coef[0] + coef[1] * eval_h
    support_resid = (eval_d - eval_expected_d - center) / scale

    return eval_e + uncertain * support_resid
