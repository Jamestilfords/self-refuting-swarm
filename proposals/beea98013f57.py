import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query assigned to an intent should point toward that intent's pooled-representation prototype, "
    "not merely be near some prototype. Classifying angular distance tails separately by classifier-selected "
    "intent detects feature-logit disagreement while accommodating intents with different within-class spread."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0032"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.pooled, dtype=float)
    xev = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]

    means = np.zeros((k, xtr.shape[1]), dtype=float)
    counts = np.bincount(y, minlength=k)
    np.add.at(means, y, xtr)
    means /= np.maximum(counts[:, None], 1)

    def angular_distance(x, prototypes):
        xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        pn = prototypes / np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
        return 1.0 - np.sum(xn * pn, axis=1)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_d = angular_distance(xtr, means[train_pred])
    eval_d = angular_distance(xev, means[eval_pred])

    scores = np.empty_like(eval_d)
    global_sorted = np.sort(train_d)
    for c in range(k):
        mask = train_pred == c
        ref = np.sort(train_d[mask]) if np.sum(mask) >= 8 else global_sorted
        take = eval_pred == c
        ranks = np.searchsorted(ref, eval_d[take], side="right")
        tail = ranks / float(ref.size + 1)
        scores[take] = eval_d[take] + tail

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
