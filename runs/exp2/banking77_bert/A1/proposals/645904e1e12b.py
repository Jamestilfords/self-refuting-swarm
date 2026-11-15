import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "The normalized shape of the Dirichlet concentration vector describes how evidence is distributed among intents independently of total evidence. OOS queries can have an atypical class-specific evidence composition even when their vacuity or energy appears in-distribution."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0017"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    z_tr = np.log(a_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(a_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]
    prototypes = np.zeros((n_classes, z_tr.shape[1]), dtype=float)
    medians = np.zeros(n_classes, dtype=float)
    scales = np.ones(n_classes, dtype=float)

    for c in range(n_classes):
        mask = labels == c
        if np.any(mask):
            proto = z_tr[mask].mean(axis=0)
            norm = np.sqrt(np.sum(proto * proto))
            if norm > eps:
                prototypes[c] = proto / norm

    tr_norm = np.sqrt(np.sum(z_tr * z_tr, axis=1))
    proto_tr = prototypes[labels]
    cosine_tr = np.sum(z_tr * proto_tr, axis=1) / np.maximum(tr_norm, eps)
    residual_tr = 1.0 - np.clip(cosine_tr, -1.0, 1.0)

    global_med = np.median(residual_tr)
    global_scale = np.median(np.abs(residual_tr - global_med)) * 1.4826 + eps
    for c in range(n_classes):
        vals = residual_tr[labels == c]
        if vals.size >= 3:
            medians[c] = np.median(vals)
            scales[c] = np.median(np.abs(vals - medians[c])) * 1.4826 + eps
        else:
            medians[c] = global_med
            scales[c] = global_scale

    pred = np.argmax(ev.logits, axis=1)
    proto_ev = prototypes[pred]
    ev_norm = np.sqrt(np.sum(z_ev * z_ev, axis=1))
    cosine_ev = np.sum(z_ev * proto_ev, axis=1) / np.maximum(ev_norm, eps)
    residual_ev = 1.0 - np.clip(cosine_ev, -1.0, 1.0)

    return (residual_ev - medians[pred]) / scales[pred]
