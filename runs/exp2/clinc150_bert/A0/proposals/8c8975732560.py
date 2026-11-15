import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "The normalized Dirichlet concentration vector describes how evidence is allocated across intents independently of total evidence. OOS queries can have an atypical evidential allocation relative to the in-scope intent selected by the classifier, producing a large Jensen-Shannon divergence."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0081"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    p_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)

    k = tr.logits.shape[1]
    global_profile = np.mean(p_tr, axis=0)
    prototypes = np.tile(global_profile, (k, 1))

    for c in range(k):
        mask = tr.labels == c
        if np.any(mask):
            prototypes[c] = np.mean(p_tr[mask], axis=0)

    prototypes = np.maximum(prototypes, eps)
    prototypes /= np.sum(prototypes, axis=1, keepdims=True)

    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)
    p_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)
    pred = np.argmax(ev.logits, axis=1)
    q = prototypes[pred]

    mid = 0.5 * (p_ev + q)
    js = 0.5 * np.sum(p_ev * np.log(p_ev / mid), axis=1)
    js += 0.5 * np.sum(q * np.log(q / mid), axis=1)
    return js
