import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "OOS representations should have low cosine similarity to every normalized in-scope "
    "class prototype, even when the classifier assigns a confident but geometrically unsupported intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalize(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    x_train = normalize(np.asarray(tr.feats, dtype=np.float64))
    x_eval = normalize(np.asarray(ev.feats, dtype=np.float64))
    labels = np.asarray(tr.labels)

    n_classes = int(np.max(labels)) + 1
    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=np.float64)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = normalize(prototypes)

    similarity = x_eval @ prototypes.T
    return 1.0 - np.max(similarity, axis=1)
