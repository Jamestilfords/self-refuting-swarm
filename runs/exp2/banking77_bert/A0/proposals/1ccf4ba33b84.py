import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Angular alignment to the prototype of the classifier-selected intent should be typical for in-scope queries. "
    "Class-conditional train calibration flags unusually weak cosine agreement without letting naturally diffuse intents dominate."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    xtr = ctx.h.l2(tr.feats)
    xev = ctx.h.l2(ev.feats)

    prototypes = np.zeros((k, xtr.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=k).astype(float)
    np.add.at(prototypes, tr.labels, xtr)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_cos = np.sum(xtr * prototypes[train_pred], axis=1)
    eval_cos = np.sum(xev * prototypes[eval_pred], axis=1)

    global_mean = float(np.mean(train_cos))
    global_std = float(np.std(train_cos))
    global_std = max(global_std, 1e-6)

    means = np.full(k, global_mean, dtype=float)
    stds = np.full(k, global_std, dtype=float)
    for c in range(k):
        vals = train_cos[train_pred == c]
        if vals.size >= 2:
            means[c] = float(np.mean(vals))
            stds[c] = max(float(np.std(vals)), 1e-6)

    return (means[eval_pred] - eval_cos) / stds[eval_pred]
