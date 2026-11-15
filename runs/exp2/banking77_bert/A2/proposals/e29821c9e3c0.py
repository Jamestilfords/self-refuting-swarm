import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentrations describe evidential allocation across intents independently "
    "of total evidence. OOS inputs can have an atypical concentration shape for their predicted intent "
    "even when vacuity or energy appears in-distribution."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0039"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)

    p_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    p_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)

    # Square-root probability vectors make cosine distance equivalent to
    # Hellinger-style discrepancy between evidential allocation patterns.
    z_tr = np.sqrt(p_tr)
    z_ev = np.sqrt(p_ev)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_proto = np.mean(z_tr, axis=0)
    global_proto /= max(np.linalg.norm(global_proto), 1e-12)

    protos = np.empty((k, z_tr.shape[1]), dtype=float)
    for c in range(k):
        members = z_tr[pred_tr == c]
        if members.shape[0] >= 2:
            proto = np.mean(members, axis=0)
            proto /= max(np.linalg.norm(proto), 1e-12)
            protos[c] = proto
        else:
            protos[c] = global_proto

    similarity = np.sum(z_ev * protos[pred_ev], axis=1)
    shape_distance = 1.0 - np.clip(similarity, -1.0, 1.0)

    # A fixed, negligible shape-only tie breaker preserves a nonconstant,
    # deterministic output in degenerate prototype configurations.
    weights = np.linspace(-1.0, 1.0, p_ev.shape[1])
    tie_break = p_ev @ weights
    return shape_distance + 1e-8 * tie_break
