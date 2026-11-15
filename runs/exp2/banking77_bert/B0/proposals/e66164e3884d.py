import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Cosine similarity to a class prototype measures whether the normalized bottleneck representation has the characteristic direction of the classifier-predicted intent. Standardizing this similarity by the predicted class's training distribution accounts for intents with intrinsically broader or tighter directional support."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0094", "m_0116"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_classes = tr.logits.shape[1]

    train_x = ctx.h.l2(tr.feats)
    eval_x = ctx.h.l2(ev.feats)

    prototypes = np.zeros((n_classes, train_x.shape[1]), dtype=float)
    means = np.zeros(n_classes, dtype=float)
    scales = np.ones(n_classes, dtype=float)

    for c in range(n_classes):
        mask = tr.labels == c
        members = train_x[mask]
        if members.shape[0] == 0:
            continue

        prototype = members.mean(axis=0)
        prototype /= max(np.linalg.norm(prototype), 1e-12)
        prototypes[c] = prototype

        train_similarity = members @ prototype
        means[c] = train_similarity.mean()
        scales[c] = max(train_similarity.std(), 1e-4)

    predicted = np.argmax(ev.logits, axis=1)
    similarity = np.sum(eval_x * prototypes[predicted], axis=1)
    return (means[predicted] - similarity) / scales[predicted]
