import numpy as np
from scipy.special import gammaln, digamma

FAMILY = "other"
ARM = "edl"
RATIONALE = "EDL queries can be out-of-scope because their full Dirichlet concentration profile is unlike the profiles produced for any predicted in-scope intent, not merely because total evidence is low. A symmetric class-conditional Dirichlet divergence captures both unusual evidence allocation across intents and anomalous concentration scale."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-6)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-6)
    k = a_tr.shape[1]

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    global_proto = np.mean(a_tr, axis=0) + 1e-4
    proto = np.empty((k, k), dtype=float)

    for c in range(k):
        members = a_tr[pred_tr == c]
        if len(members) >= 2:
            proto[c] = np.mean(members, axis=0) + 1e-4
        else:
            proto[c] = global_proto

    b = proto[pred_ev]

    sum_a = np.sum(a_ev, axis=1)
    sum_b = np.sum(b, axis=1)

    kl_ab = (
        gammaln(sum_a)
        - np.sum(gammaln(a_ev), axis=1)
        - gammaln(sum_b)
        + np.sum(gammaln(b), axis=1)
        + np.sum((a_ev - b) * (digamma(a_ev) - digamma(sum_a)[:, None]), axis=1)
    )

    kl_ba = (
        gammaln(sum_b)
        - np.sum(gammaln(b), axis=1)
        - gammaln(sum_a)
        + np.sum(gammaln(a_ev), axis=1)
        + np.sum((b - a_ev) * (digamma(b) - digamma(sum_b)[:, None]), axis=1)
    )

    return np.maximum(0.5 * (kl_ab + kl_ba), 0.0)
