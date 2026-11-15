import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration vector describes how evidence is distributed across intents, "
    "not merely how much total evidence exists. OOS queries often have an evidence-allocation shape that "
    "mismatches the class-conditional shapes learned from in-scope examples."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)
    p_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    p_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)

    k = p_tr.shape[1]
    labels = np.asarray(tr.labels, dtype=int)
    counts = np.bincount(labels, minlength=k).astype(float)
    sums = np.zeros((k, k), dtype=float)
    np.add.at(sums, labels, p_tr)

    global_proto = np.mean(p_tr, axis=0)
    prototypes = (sums + global_proto[None, :]) / (counts[:, None] + 1.0)
    prototypes /= np.sum(prototypes, axis=1, keepdims=True)

    pred = np.argmax(ev.logits, axis=1)
    q = prototypes[pred]
    m = 0.5 * (p_ev + q)

    js = 0.5 * np.sum(
        p_ev * (np.log(p_ev + eps) - np.log(m + eps)), axis=1
    ) + 0.5 * np.sum(
        q * (np.log(q + eps) - np.log(m + eps)), axis=1
    )
    return js
