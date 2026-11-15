import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "L2-normalized class prototypes measure angular agreement with known intents while removing feature-norm variation. Queries with low maximum cosine similarity to every training class prototype receive higher OOS scores."
SOURCE = "https://github.com/pradhyumnaag30/CLINC150"
PARENTS = []
MEMORY_USED = ["m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels)
    classes = np.unique(labels)

    prototypes = np.empty((len(classes), x_train.shape[1]), dtype=float)
    for i, c in enumerate(classes):
        proto = x_train[labels == c].mean(axis=0)
        norm = np.linalg.norm(proto)
        prototypes[i] = proto / max(norm, 1e-12)

    max_similarity = np.max(x_eval @ prototypes.T, axis=1)
    return np.clip(1.0 - max_similarity, 0.0, 2.0)
