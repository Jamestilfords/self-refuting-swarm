import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Each intent occupies a characteristic direction in normalized bottleneck space. "
    "A query whose classifier-assigned intent has unusually low cosine similarity to its class prototype is likely OOS."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0060"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalize(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    xtr = normalize(np.asarray(tr.feats, dtype=float))
    xev = normalize(np.asarray(ev.feats, dtype=float))
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    prototypes = np.zeros((n_classes, xtr.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, xtr)
    prototypes = normalize(prototypes / np.maximum(counts[:, None], 1.0))

    train_cos = np.sum(xtr * prototypes[labels], axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    eval_cos = np.sum(xev * prototypes[eval_pred], axis=1)

    score_out = np.empty(len(xev), dtype=float)
    for c in range(n_classes):
        mask = eval_pred == c
        reference = np.sort(train_cos[labels == c])
        if reference.size == 0:
            score_out[mask] = 1.0 - eval_cos[mask]
        else:
            percentile = np.searchsorted(reference, eval_cos[mask], side="right")
            percentile = percentile / float(reference.size)
            score_out[mask] = 1.0 - percentile

    return score_out
