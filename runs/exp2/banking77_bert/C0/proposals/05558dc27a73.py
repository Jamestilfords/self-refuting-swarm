import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples tend to produce class-specific relative logit patterns, not merely a high winning logit. OOS queries can have a plausible margin but an atypical full centered-logit direction, yielding low similarity to every training intent prototype."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalize_shape(x):
        z = np.asarray(x, dtype=float)
        z = z - np.mean(z, axis=1, keepdims=True)
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)

    train_shape = normalize_shape(tr.logits)
    eval_shape = normalize_shape(ev.logits)
    labels = np.asarray(tr.labels)

    classes = np.unique(labels)
    prototypes = np.empty((len(classes), train_shape.shape[1]), dtype=float)
    for i, c in enumerate(classes):
        prototype = np.mean(train_shape[labels == c], axis=0)
        prototypes[i] = prototype / max(np.linalg.norm(prototype), 1e-12)

    similarity = eval_shape @ prototypes.T
    return 1.0 - np.max(similarity, axis=1)
