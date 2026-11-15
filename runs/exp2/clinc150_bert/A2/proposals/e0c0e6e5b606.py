import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Known queries should have cosine similarity to their classifier-predicted intent "
    "prototype that exceeds similarity to competing intent prototypes. A low, "
    "class-calibrated prototype-agreement margin indicates semantic mismatch with the assigned intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0099"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    labels = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    dim = x_train.shape[1]

    prototypes = np.zeros((k, dim), dtype=float)
    counts = np.bincount(labels, minlength=k).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_sim = x_train @ prototypes.T
    own_train = train_sim[np.arange(labels.size), labels]
    train_sim[np.arange(labels.size), labels] = -np.inf
    rival_train = np.max(train_sim, axis=1)
    train_margin = own_train - rival_train

    sums = np.bincount(labels, weights=train_margin, minlength=k)
    means = sums / np.maximum(counts, 1.0)
    centered = train_margin - means[labels]
    vars_ = np.bincount(labels, weights=centered * centered, minlength=k)
    stds = np.sqrt(vars_ / np.maximum(counts, 1.0))
    stds = np.maximum(stds, 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    eval_sim = x_eval @ prototypes.T
    own_eval = eval_sim[np.arange(pred.size), pred]
    eval_sim[np.arange(pred.size), pred] = -np.inf
    rival_eval = np.max(eval_sim, axis=1)
    eval_margin = own_eval - rival_eval

    return np.clip((means[pred] - eval_margin) / stds[pred], -1e6, 1e6)
