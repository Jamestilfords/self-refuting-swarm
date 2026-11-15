import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = (
    "Dirichlet mean vectors encode the relative allocation of evidence across intents, "
    "independently of total evidence (vacuity). A query whose evidential distribution is "
    "unusually dissimilar to the training evidence profile of its predicted intent is likely OOS."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0047"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-10

    alpha_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    alpha_ev = np.maximum(ctx.h.alpha(ev.logits), eps)
    q_tr = alpha_tr / np.maximum(alpha_tr.sum(axis=1, keepdims=True), eps)
    q_ev = alpha_ev / np.maximum(alpha_ev.sum(axis=1, keepdims=True), eps)

    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]
    global_proto = (q_tr.sum(axis=0) + 1.0) / (len(q_tr) + n_classes)

    proto = np.empty((n_classes, q_tr.shape[1]), dtype=float)
    for c in range(n_classes):
        mask = labels == c
        count = int(mask.sum())
        if count:
            # Shrink sparse intent profiles toward the global evidential profile.
            weight = count / (count + 8.0)
            proto[c] = weight * q_tr[mask].mean(axis=0) + (1.0 - weight) * global_proto
        else:
            proto[c] = global_proto

    proto = np.maximum(proto, eps)
    proto /= proto.sum(axis=1, keepdims=True)

    def js_divergence(q, p):
        m = 0.5 * (q + p)
        return 0.5 * (
            np.sum(q * (np.log(q) - np.log(m)), axis=1)
            + np.sum(p * (np.log(p) - np.log(m)), axis=1)
        )

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    raw_tr = js_divergence(q_tr, proto[pred_tr])
    raw_ev = js_divergence(q_ev, proto[pred_ev])

    # Predicted-intent conditional calibration removes naturally diffuse intents.
    centers = np.empty(n_classes, dtype=float)
    scales = np.empty(n_classes, dtype=float)
    global_center = float(np.median(raw_tr))
    global_scale = float(np.median(np.abs(raw_tr - global_center)) * 1.4826 + eps)

    for c in range(n_classes):
        vals = raw_tr[pred_tr == c]
        if len(vals) >= 5:
            center = float(np.median(vals))
            scale = float(np.median(np.abs(vals - center)) * 1.4826)
            centers[c] = center
            scales[c] = max(scale, 0.1 * global_scale, eps)
        else:
            centers[c] = global_center
            scales[c] = global_scale

    return (raw_ev - centers[pred_ev]) / scales[pred_ev]
