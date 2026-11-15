import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "L2-normalized class prototypes capture the directional center of each in-scope intent cluster. OOS queries should have lower maximum cosine similarity to every training-derived prototype."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    sums = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(sums, labels, x_train)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    prototypes = sums / np.maximum(counts[:, None], 1.0)

    proto_norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes = prototypes / np.maximum(proto_norms, 1e-12)

    eval_norms = np.linalg.norm(x_eval, axis=1, keepdims=True)
    x_eval = x_eval / np.maximum(eval_norms, 1e-12)

    similarities = x_eval @ prototypes.T
    return 1.0 - np.max(similarities, axis=1)
