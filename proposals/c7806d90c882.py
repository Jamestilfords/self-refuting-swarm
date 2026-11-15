import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A valid predicted intent should have both a typical class-conditional Mahalanobis distance and a clear geometric advantage over competing intent prototypes. Combining predicted-class distance surprise with prototype-boundary ambiguity exposes feature-logit disagreement without reducing to nearest-class density alone."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0032", "m_0018"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    d_tr = g.class_mahalanobis_sq(tr.pooled)
    d_ev = g.class_mahalanobis_sq(ev.pooled)

    k = d_tr.shape[1]
    pred_tr = np.clip(np.argmax(tr.logits, axis=1), 0, k - 1)
    pred_ev = np.clip(np.argmax(ev.logits, axis=1), 0, k - 1)

    row_tr = np.arange(d_tr.shape[0])
    row_ev = np.arange(d_ev.shape[0])
    selected_tr = d_tr[row_tr, pred_tr]
    selected_ev = d_ev[row_ev, pred_ev]

    global_med = float(np.median(selected_tr))
    global_scale = float(np.percentile(selected_tr, 90) - np.percentile(selected_tr, 50))
    global_scale = max(global_scale, 1e-6)

    med = np.full(k, global_med, dtype=float)
    scale = np.full(k, global_scale, dtype=float)
    for c in range(k):
        values = selected_tr[pred_tr == c]
        if values.size >= 8:
            med[c] = np.median(values)
            spread = np.percentile(values, 90) - np.percentile(values, 50)
            scale[c] = max(float(spread), 1e-6)

    distance_surprise = (selected_ev - med[pred_ev]) / scale[pred_ev]

    if k > 1:
        near_tr = np.partition(d_tr, 1, axis=1)[:, :2]
        near_ev = np.partition(d_ev, 1, axis=1)[:, :2]
        ambiguity_tr = near_tr[:, 0] / (near_tr[:, 1] + 1e-6)
        ambiguity_ev = near_ev[:, 0] / (near_ev[:, 1] + 1e-6)

        amb_med = float(np.median(ambiguity_tr))
        amb_scale = float(
            np.percentile(ambiguity_tr, 90) - np.percentile(ambiguity_tr, 50)
        )
        amb_scale = max(amb_scale, 1e-6)
        ambiguity_surprise = (ambiguity_ev - amb_med) / amb_scale
    else:
        ambiguity_surprise = np.zeros_like(distance_surprise)

    scores = distance_surprise + 0.35 * ambiguity_surprise
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
