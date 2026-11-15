import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "In-scope EDL examples should produce class-specific Dirichlet concentration profiles, not merely high total evidence. This score detects examples whose normalized concentration shape and total concentration are atypical for every training class."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    p_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    p_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)
    log_s_tr = np.log(np.sum(a_tr, axis=1))
    log_s_ev = np.log(np.sum(a_ev, axis=1))

    k = a_tr.shape[1]
    global_proto = np.mean(p_tr, axis=0)
    global_mean = np.mean(log_s_tr)
    global_std = np.std(log_s_tr) + 1e-6

    proto = np.empty((k, k), dtype=float)
    means = np.empty(k, dtype=float)
    stds = np.empty(k, dtype=float)
    for c in range(k):
        mask = tr.labels == c
        if np.any(mask):
            proto[c] = np.mean(p_tr[mask], axis=0)
            means[c] = np.mean(log_s_tr[mask])
            stds[c] = np.std(log_s_tr[mask]) + 1e-6
        else:
            proto[c] = global_proto
            means[c] = global_mean
            stds[c] = global_std

    proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-8)
    p_ev /= np.maximum(np.linalg.norm(p_ev, axis=1, keepdims=True), 1e-8)

    shape_match = p_ev @ proto.T
    evidence_deviation = np.abs((log_s_ev[:, None] - means[None, :]) / stds[None, :])
    compatibility = shape_match - 0.15 * evidence_deviation
    return 1.0 - np.max(compatibility, axis=1)
