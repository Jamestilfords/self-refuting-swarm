import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query can be close to some class while still being atypical for the class selected by the classifier. Class-conditional calibration of the predicted-class Mahalanobis distance highlights such classifier-density disagreements."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    tr_d = g.class_mahalanobis_sq(tr.pooled)
    ev_d = g.class_mahalanobis_sq(ev.pooled)

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    n_classes = tr_d.shape[1]
    tr_pred = np.clip(tr_pred, 0, n_classes - 1)
    ev_pred = np.clip(ev_pred, 0, n_classes - 1)

    tr_selected = tr_d[np.arange(tr_d.shape[0]), tr_pred]
    ev_selected = ev_d[np.arange(ev_d.shape[0]), ev_pred]

    global_med = np.median(tr_selected)
    global_scale = np.median(np.abs(tr_selected - global_med)) * 1.4826
    global_scale = max(float(global_scale), 1e-6)

    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = tr_selected[tr_pred == c]
        if values.size >= 5:
            med = np.median(values)
            scale = np.median(np.abs(values - med)) * 1.4826
            medians[c] = med
            scales[c] = max(float(scale), 1e-6)

    scores = (ev_selected - medians[ev_pred]) / scales[ev_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
