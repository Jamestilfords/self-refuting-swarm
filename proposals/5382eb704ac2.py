import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The centered log-ratio shape of Dirichlet concentrations describes relative evidence allocation independently of total evidence. "
    "Conditioning this shape on predicted intent and total concentration flags atypical evidential competition patterns without being a monotone transform of energy or vacuity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0038"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]
    n_dims = a_tr.shape[1]

    log_total_tr = np.log(np.sum(a_tr, axis=1))
    log_total_ev = np.log(np.sum(a_ev, axis=1))

    clr_tr = np.log(a_tr)
    clr_tr -= np.mean(clr_tr, axis=1, keepdims=True)
    clr_ev = np.log(a_ev)
    clr_ev -= np.mean(clr_ev, axis=1, keepdims=True)

    global_mean = np.mean(clr_tr, axis=0)
    global_x = np.mean(log_total_tr)
    global_xc = log_total_tr - global_x
    global_denom = np.sum(global_xc * global_xc) + 1e-8
    global_slope = (global_xc[:, None] * (clr_tr - global_mean)).sum(axis=0) / global_denom
    global_resid = clr_tr - global_mean - global_xc[:, None] * global_slope
    var = np.mean(global_resid * global_resid, axis=0) + 1e-6

    means = np.tile(global_mean, (n_classes, 1))
    slopes = np.tile(global_slope, (n_classes, 1))
    x_means = np.full(n_classes, global_x)

    for c in range(n_classes):
        mask = pred_tr == c
        if np.sum(mask) < 4:
            continue
        x = log_total_tr[mask]
        y = clr_tr[mask]
        xm = np.mean(x)
        ym = np.mean(y, axis=0)
        xc = x - xm
        denom = np.sum(xc * xc)
        if denom > 1e-8:
            means[c] = ym
            slopes[c] = (xc[:, None] * (y - ym)).sum(axis=0) / denom
            x_means[c] = xm

    residual = clr_ev - means[pred_ev] - (
        log_total_ev - x_means[pred_ev]
    )[:, None] * slopes[pred_ev]

    return np.mean((residual * residual) / var, axis=1)
