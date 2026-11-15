import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query may be close to some intent prototype yet disagree geometrically with the intent selected by the classifier. Class-conditional calibration of cosine agreement between normalized bottleneck features and the predicted intent prototype flags such logit-representation mismatches."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = ctx.h.l2(tr.feats)
    xev = ctx.h.l2(ev.feats)

    k = tr.logits.shape[1]
    d = xtr.shape[1]
    prototypes = np.zeros((k, d), dtype=float)
    counts = np.bincount(tr.labels, minlength=k).astype(float)
    np.add.at(prototypes, tr.labels, xtr)

    global_proto = np.mean(xtr, axis=0, keepdims=True)
    missing = counts == 0
    if np.any(missing):
        prototypes[missing] = global_proto
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_sim = np.sum(xtr * prototypes[train_pred], axis=1)
    eval_sim = np.sum(xev * prototypes[eval_pred], axis=1)

    scores = np.empty(len(eval_sim), dtype=float)
    for c in range(k):
        mask_train = train_pred == c
        mask_eval = eval_pred == c
        reference = train_sim[mask_train]
        if reference.size < 4:
            reference = train_sim[tr.labels == c]
        if reference.size == 0:
            reference = train_sim
        reference = np.sort(reference)
        rank = np.searchsorted(reference, eval_sim[mask_eval], side="right")
        low_tail = rank / float(reference.size)
        scores[mask_eval] = 1.0 - low_tail + 1e-3 * (1.0 - eval_sim[mask_eval])

    return scores
