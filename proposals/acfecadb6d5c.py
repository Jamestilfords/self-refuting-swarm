import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Maximum-softmax uncertainty captures classifier ambiguity, while classifier-predicted bottleneck Mahalanobis distance tests whether that decision has class-conditional representation support. Train empirical tail maps place both signals on a shared outlier-probability scale and reward agreement between them."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0021", "m_0037", "m_0066"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", pca=None, whiten=True)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_all_d = g.class_mahalanobis_sq(tr.feats)
    eval_all_d = g.class_mahalanobis_sq(ev.feats)
    train_d = train_all_d[np.arange(len(tr.labels)), train_pred]
    eval_d = eval_all_d[np.arange(len(eval_pred)), eval_pred]

    train_msp = ctx.h.msp(tr.logits)
    eval_msp = ctx.h.msp(ev.logits)

    def upper_tail(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=float))
        return np.searchsorted(ref, values, side="right") / float(len(ref) + 1)

    msp_tail = upper_tail(train_msp, eval_msp)
    dist_tail = upper_tail(train_d, eval_d)

    score_values = np.sqrt(np.maximum(msp_tail, 1e-8) * np.maximum(dist_tail, 1e-8))
    return np.nan_to_num(score_values, nan=0.0, posinf=1.0, neginf=0.0)
