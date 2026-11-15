import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Maximum softmax uncertainty and bottleneck-space distance capture complementary classifier and representation novelty signals. Robust train-fitted median/IQR scaling places both signals on a common bounded scale without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0105"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    train_msp = ctx.h.msp(tr.logits)
    train_dist = g.min_mahalanobis_sq(tr.feats)

    def robust_sigmoid(x, reference):
        med = np.median(reference)
        q25, q75 = np.percentile(reference, [25.0, 75.0])
        scale = max(float(q75 - q25), 1e-6)
        z = np.clip((x - med) / scale, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    msp_score = robust_sigmoid(ctx.h.msp(ev.logits), train_msp)
    dist_score = robust_sigmoid(g.min_mahalanobis_sq(ev.feats), train_dist)
    return msp_score + dist_score
