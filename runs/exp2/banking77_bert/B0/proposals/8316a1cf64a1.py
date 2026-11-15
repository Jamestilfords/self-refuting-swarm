import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "Removing total Dirichlet evidence leaves a concentration-shape signature that should be "
    "consistent within each predicted intent; angular deviation from that intent's training shape prototype flags atypical evidential allocations."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0098", "m_0115"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    z_tr = np.log(a_tr)
    z_ev = np.log(a_ev)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    z_tr = z_tr / np.maximum(np.linalg.norm(z_tr, axis=1, keepdims=True), eps)
    z_ev = z_ev / np.maximum(np.linalg.norm(z_ev, axis=1, keepdims=True), eps)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_proto = z_tr.mean(axis=0)
    global_proto /= max(np.linalg.norm(global_proto), eps)

    protos = np.empty((k, z_tr.shape[1]), dtype=float)
    for c in range(k):
        members = z_tr[pred_tr == c]
        if len(members) == 0:
            protos[c] = global_proto
        else:
            p = members.mean(axis=0)
            protos[c] = p / max(np.linalg.norm(p), eps)

    similarity = np.sum(z_ev * protos[pred_ev], axis=1)
    return np.clip(1.0 - similarity, 0.0, 2.0)
