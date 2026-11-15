import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope examples assigned to an intent usually have a class-conditionally typical "
    "Mahalanobis separation between their nearest and second-nearest intent prototypes. "
    "Low prototype isolation indicates ambiguous geometric support even when the classifier is confident."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0176"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    d_tr = g.class_mahalanobis_sq(tr.pooled)
    d_ev = g.class_mahalanobis_sq(ev.pooled)

    tr_sorted = np.partition(d_tr, 1, axis=1)
    ev_sorted = np.partition(d_ev, 1, axis=1)

    scale = np.median(tr_sorted[:, 0]) + 1e-6
    gap_tr = (tr_sorted[:, 1] - tr_sorted[:, 0]) / (tr_sorted[:, 0] + scale)
    gap_ev = (ev_sorted[:, 1] - ev_sorted[:, 0]) / (ev_sorted[:, 0] + scale)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_mean = float(np.mean(gap_tr))
    global_std = float(np.std(gap_tr) + 1e-6)
    means = np.full(k, global_mean, dtype=float)
    stds = np.full(k, global_std, dtype=float)

    for c in range(k):
        values = gap_tr[pred_tr == c]
        if values.size >= 2:
            means[c] = np.mean(values)
            stds[c] = max(float(np.std(values)), 1e-6)

    scores = (means[pred_ev] - gap_ev) / stds[pred_ev]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
