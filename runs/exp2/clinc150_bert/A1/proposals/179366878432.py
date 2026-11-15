import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-conditional cosine similarity measures angular compatibility with the intent predicted by "
    "the classifier. Calibrating similarity against that intent's in-scope training tail flags examples "
    "whose representation is atypical for their assigned class."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels, dtype=int)

    n_classes = max(
        int(np.max(labels)) + 1,
        tr.logits.shape[1],
        ev.logits.shape[1],
    )
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_similarity = np.sum(x_train * prototypes[labels], axis=1)
    predicted = np.argmax(ev.logits, axis=1)
    eval_similarity = np.sum(x_eval * prototypes[predicted], axis=1)

    score_out = np.empty(eval_similarity.shape[0], dtype=float)
    for c in range(n_classes):
        mask = predicted == c
        if not np.any(mask):
            continue
        reference = np.sort(train_similarity[labels == c])
        if reference.size == 0:
            reference = np.sort(train_similarity)
        percentile = np.searchsorted(reference, eval_similarity[mask], side="right")
        score_out[mask] = 1.0 - percentile / (reference.size + 1.0)

    return score_out
