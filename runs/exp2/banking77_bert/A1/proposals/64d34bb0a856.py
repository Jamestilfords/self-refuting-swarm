import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Maximum softmax uncertainty and bottleneck-space Gaussian distance capture complementary classifier and representation novelty. Train empirical CDF calibration puts both signals on a shared robust percentile scale without inspecting evaluation-set statistics."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0063"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    train_msp = np.nan_to_num(ctx.h.msp(tr.logits), nan=0.0, posinf=1e6, neginf=-1e6)
    eval_msp = np.nan_to_num(ctx.h.msp(ev.logits), nan=0.0, posinf=1e6, neginf=-1e6)

    train_dist = np.nan_to_num(
        g.min_mahalanobis_sq(tr.feats), nan=0.0, posinf=1e6, neginf=-1e6
    )
    eval_dist = np.nan_to_num(
        g.min_mahalanobis_sq(ev.feats), nan=0.0, posinf=1e6, neginf=-1e6
    )

    msp_ref = np.sort(train_msp)
    dist_ref = np.sort(train_dist)
    n = float(max(len(tr.labels), 1))

    msp_pct = np.searchsorted(msp_ref, eval_msp, side="right") / n
    dist_pct = np.searchsorted(dist_ref, eval_dist, side="right") / n

    return 0.5 * msp_pct + 0.5 * dist_pct
