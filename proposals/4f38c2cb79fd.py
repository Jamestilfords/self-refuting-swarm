import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Classifier confidence should imply a characteristic amount of representation support from the "
    "predicted intent. A pooled-space predicted-class Mahalanobis distance that is unexpectedly large "
    "conditional on the logit margin exposes classifier-feature disagreement for OOS queries."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0032", "m_0153"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    d_train_all = g.class_mahalanobis_sq(tr.pooled)
    d_eval_all = g.class_mahalanobis_sq(ev.pooled)

    def margin(logits):
        k = logits.shape[1]
        top2 = np.partition(logits, k - 2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    row_train = np.arange(pred_train.shape[0])
    row_eval = np.arange(pred_eval.shape[0])

    y = d_train_all[row_train, pred_train]
    x = margin(tr.logits)
    x_eval = margin(ev.logits)

    xm = float(np.mean(x))
    ym = float(np.mean(y))
    xc = x - xm
    denom = float(np.dot(xc, xc)) + 1e-6
    global_slope = float(np.dot(xc, y - ym) / denom)
    global_intercept = ym - global_slope * xm
    global_resid = y - (global_intercept + global_slope * x)
    global_scale = max(float(np.std(global_resid)), 1e-3)

    n_classes = d_train_all.shape[1]
    intercepts = np.full(n_classes, global_intercept, dtype=float)
    slopes = np.full(n_classes, global_slope, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        mask = pred_train == c
        n = int(np.sum(mask))
        if n < 3:
            continue

        xc_c = x[mask]
        yc_c = y[mask]
        mx = float(np.mean(xc_c))
        my = float(np.mean(yc_c))
        centered_x = xc_c - mx
        slope_c = float(
            np.dot(centered_x, yc_c - my) /
            (np.dot(centered_x, centered_x) + 1e-4)
        )
        intercept_c = my - slope_c * mx

        weight = n / (n + 20.0)
        intercepts[c] = weight * intercept_c + (1.0 - weight) * global_intercept
        slopes[c] = weight * slope_c + (1.0 - weight) * global_slope

        resid_c = yc_c - (intercepts[c] + slopes[c] * xc_c)
        scale_c = float(np.std(resid_c))
        scales[c] = max(
            weight * scale_c + (1.0 - weight) * global_scale,
            1e-3,
        )

    d_eval = d_eval_all[row_eval, pred_eval]
    expected = intercepts[pred_eval] + slopes[pred_eval] * x_eval
    scores = (d_eval - expected) / scales[pred_eval]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
