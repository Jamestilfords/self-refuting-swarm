import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "OOS inputs can induce an unusual allocation of evidential mass across classes even when total "
    "Dirichlet strength is not low. A predicted-class-conditioned surprise score on centered log "
    "concentrations detects atypical Dirichlet shapes independently of vacuity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    # Centering log concentrations removes the overall evidence-strength component,
    # leaving only the relative shape of the Dirichlet concentration vector.
    x_tr = np.log(a_tr)
    x_ev = np.log(a_ev)
    x_tr = x_tr - x_tr.mean(axis=1, keepdims=True)
    x_ev = x_ev - x_ev.mean(axis=1, keepdims=True)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_center = np.median(x_tr, axis=0)
    global_scale = np.median(np.abs(x_tr - global_center), axis=0)
    global_scale = np.maximum(1.4826 * global_scale, 1e-3)

    centers = np.empty((n_classes, x_tr.shape[1]), dtype=float)
    scales = np.empty_like(centers)

    for c in range(n_classes):
        members = x_tr[pred_tr == c]
        if members.shape[0] >= 8:
            center = np.median(members, axis=0)
            scale = np.median(np.abs(members - center), axis=0)
            centers[c] = center
            scales[c] = np.maximum(1.4826 * scale, 1e-3)
        else:
            centers[c] = global_center
            scales[c] = global_scale

    z = (x_ev - centers[pred_ev]) / scales[pred_ev]
    return np.mean(np.minimum(z * z, 100.0), axis=1)
