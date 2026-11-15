import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "L2-normalized encoder features represent directional semantics, so OOS queries should have lower cosine similarity to every class prototype than in-scope queries. Calibrating the nearest-prototype cosine distance using training scores avoids evaluation-set normalization."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0007"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    labels = tr.labels
    classes = np.unique(labels)

    prototypes = np.empty((len(classes), x_train.shape[1]), dtype=float)
    for i, cls in enumerate(classes):
        proto = np.mean(x_train[labels == cls], axis=0)
        norm = np.linalg.norm(proto)
        prototypes[i] = proto / max(norm, 1e-12)

    train_best = np.max(x_train @ prototypes.T, axis=1)
    eval_best = np.max(x_eval @ prototypes.T, axis=1)

    center = np.median(train_best)
    scale = np.median(np.abs(train_best - center))
    scale = max(float(scale), 1e-6)
    scores = (center - eval_best) / scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
