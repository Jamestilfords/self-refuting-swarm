import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query assigned to an intent should be close to that intent's Gaussian mean, and that assigned intent should not be much worse than the closest feature-space intent. Class-conditioned calibration of predicted-class novelty and classifier-feature disagreement highlights atypical assignments without using evaluation-set statistics."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0029"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    d_tr = g.class_mahalanobis_sq(tr.pooled)
    d_ev = g.class_mahalanobis_sq(ev.pooled)
    n_classes = d_tr.shape[1]

    p_tr = np.clip(np.argmax(tr.logits, axis=1), 0, n_classes - 1)
    p_ev = np.clip(np.argmax(ev.logits, axis=1), 0, n_classes - 1)

    rows_tr = np.arange(d_tr.shape[0])
    rows_ev = np.arange(d_ev.shape[0])

    pred_tr = d_tr[rows_tr, p_tr]
    pred_ev = d_ev[rows_ev, p_ev]
    near_tr = np.min(d_tr, axis=1)
    near_ev = np.min(d_ev, axis=1)

    gap_tr = pred_tr - near_tr
    gap_ev = pred_ev - near_ev

    pred_center = np.zeros(n_classes)
    pred_scale = np.ones(n_classes)
    gap_center = np.zeros(n_classes)
    gap_scale = np.ones(n_classes)

    global_pred_center = np.median(pred_tr)
    global_pred_scale = max(
        np.quantile(pred_tr, 0.75) - np.quantile(pred_tr, 0.25), 1e-6
    )
    global_gap_center = np.median(gap_tr)
    global_gap_scale = max(
        np.quantile(gap_tr, 0.75) - np.quantile(gap_tr, 0.25), 1e-6
    )

    for c in range(n_classes):
        mask = p_tr == c
        if np.sum(mask) >= 4:
            pc = pred_tr[mask]
            gc = gap_tr[mask]
            pred_center[c] = np.median(pc)
            pred_scale[c] = max(
                np.quantile(pc, 0.75) - np.quantile(pc, 0.25), 1e-6
            )
            gap_center[c] = np.median(gc)
            gap_scale[c] = max(
                np.quantile(gc, 0.75) - np.quantile(gc, 0.25), 1e-6
            )
        else:
            pred_center[c] = global_pred_center
            pred_scale[c] = global_pred_scale
            gap_center[c] = global_gap_center
            gap_scale[c] = global_gap_scale

    novelty = (pred_ev - pred_center[p_ev]) / pred_scale[p_ev]
    disagreement = (gap_ev - gap_center[p_ev]) / gap_scale[p_ev]
    return novelty + 0.5 * disagreement
