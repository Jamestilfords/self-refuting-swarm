import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration vector describes how evidence is allocated among intents, "
    "independently of its total magnitude. Class-conditional deviations in its concentration entropy flag "
    "queries whose evidential shape is atypical for the intent selected by the classifier."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0072"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)

    q_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    q_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)

    ent_tr = -np.sum(q_tr * np.log(q_tr), axis=1)
    ent_ev = -np.sum(q_ev * np.log(q_ev), axis=1)

    k = tr.logits.shape[1]
    means = np.zeros(k, dtype=float)
    scales = np.zeros(k, dtype=float)

    global_mean = float(np.mean(ent_tr))
    global_scale = float(np.std(ent_tr))
    global_scale = max(global_scale, 1e-6)

    for c in range(k):
        vals = ent_tr[tr.labels == c]
        if vals.size >= 2:
            means[c] = np.mean(vals)
            scales[c] = max(float(np.std(vals)), 1e-6)
        else:
            means[c] = global_mean
            scales[c] = global_scale

    pred = np.argmax(ev.logits, axis=1)
    z = (ent_ev - means[pred]) / scales[pred]
    return z * z
