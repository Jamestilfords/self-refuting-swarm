import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "OOS inputs can produce an evidence composition across classes that is atypical for the class they appear to predict. This score measures class-conditional deviation in centered log-Dirichlet concentration space, removing overall concentration so it is not a monotone transform of energy or vacuity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    z_tr = np.log(a_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(a_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    k = z_tr.shape[1]
    means = np.zeros((k, k), dtype=float)
    counts = np.bincount(tr.labels, minlength=k).astype(float)

    for c in range(k):
        mask = tr.labels == c
        if np.any(mask):
            means[c] = z_tr[mask].mean(axis=0)

    global_mean = z_tr.mean(axis=0)
    means[counts == 0] = global_mean

    residual = z_tr - means[tr.labels]
    var = np.mean(residual * residual, axis=0)
    floor = max(float(np.mean(var)) * 1e-3, 1e-6)
    var = np.maximum(var, floor)

    pred = np.argmax(a_ev, axis=1)
    delta = z_ev - means[pred]
    shape_distance = np.mean((delta * delta) / var, axis=1)

    q_ev = a_ev / a_ev.sum(axis=1, keepdims=True)
    proto_logits = means[pred]
    proto_logits = proto_logits - proto_logits.max(axis=1, keepdims=True)
    proto = np.exp(proto_logits)
    proto /= proto.sum(axis=1, keepdims=True)

    disagreement = np.sum(q_ev * (np.log(q_ev + 1e-12) - np.log(proto + 1e-12)), axis=1)
    return shape_distance + disagreement
