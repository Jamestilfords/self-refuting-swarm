import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "For each classifier-selected intent, in-scope representations have a characteristic "
    "relationship between logit confidence and distance from that intent's feature prototype. "
    "A large class-conditional distance residual indicates that the predicted intent is not "
    "geometrically supported even after accounting for decision confidence."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0032", "m_0288", "m_0293"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    def margin_and_prediction(logits):
        pred = np.argmax(logits, axis=1)
        if logits.shape[1] < 2:
            margin = np.zeros(logits.shape[0], dtype=float)
        else:
            top2 = np.partition(logits, -2, axis=1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
        return pred, np.log1p(np.maximum(margin, 0.0))

    pred_tr, conf_tr = margin_and_prediction(tr.logits)
    pred_ev, conf_ev = margin_and_prediction(ev.logits)

    dtr_all = g.class_mahalanobis_sq(tr.pooled)
    dev_all = g.class_mahalanobis_sq(ev.pooled)
    n_classes = dtr_all.shape[1]

    pred_tr = np.clip(pred_tr, 0, n_classes - 1)
    pred_ev = np.clip(pred_ev, 0, n_classes - 1)
    dtr = dtr_all[np.arange(dtr_all.shape[0]), pred_tr]
    dev = dev_all[np.arange(dev_all.shape[0]), pred_ev]

    def fit_line(x, y):
        xm = float(np.mean(x))
        ym = float(np.mean(y))
        var = float(np.mean((x - xm) ** 2))
        slope = float(np.mean((x - xm) * (y - ym)) / (var + 1e-6))
        intercept = ym - slope * xm
        residual = y - intercept - slope * x
        scale = float(np.sqrt(np.mean(residual ** 2)) + 1e-6)
        return intercept, slope, scale

    global_intercept, global_slope, global_scale = fit_line(conf_tr, dtr)
    intercept = np.full(n_classes, global_intercept, dtype=float)
    slope = np.full(n_classes, global_slope, dtype=float)
    scale = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        mask = pred_tr == c
        if np.sum(mask) >= 8:
            intercept[c], slope[c], scale[c] = fit_line(conf_tr[mask], dtr[mask])

    out = (dev - intercept[pred_ev] - slope[pred_ev] * conf_ev) / scale[pred_ev]
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
