import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Cosine similarity to the prototype of the classifier-predicted intent measures angular feature "
    "support, while class-conditional train calibration avoids penalizing naturally broad intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0020"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = ctx.h.l2(tr.feats)
    xev = ctx.h.l2(ev.feats)

    k = tr.logits.shape[1]
    d = xtr.shape[1]
    labels = tr.labels.astype(int)

    global_proto = ctx.h.l2(np.mean(xtr, axis=0, keepdims=True))[0]
    prototypes = np.tile(global_proto, (k, 1))
    counts = np.bincount(labels, minlength=k)

    for c in range(k):
        if counts[c] > 0:
            prototypes[c] = ctx.h.l2(
                np.mean(xtr[labels == c], axis=0, keepdims=True)
            )[0]

    pred_eval = np.argmax(ev.logits, axis=1)
    eval_sim = np.sum(xev * prototypes[pred_eval], axis=1)

    train_sim_by_class = []
    all_train_sim = np.sum(xtr * prototypes[np.clip(labels, 0, k - 1)], axis=1)
    fallback = np.sort(all_train_sim)

    for c in range(k):
        if counts[c] > 1:
            sims = np.sum(xtr[labels == c] * prototypes[c], axis=1)
            train_sim_by_class.append(np.sort(sims))
        else:
            train_sim_by_class.append(fallback)

    scores = np.empty(len(xev), dtype=float)
    for c in range(k):
        mask = pred_eval == c
        if np.any(mask):
            reference = train_sim_by_class[c]
            rank = np.searchsorted(reference, eval_sim[mask], side="right")
            support = (rank + 1.0) / (reference.size + 2.0)
            scores[mask] = 1.0 - support

    return scores
