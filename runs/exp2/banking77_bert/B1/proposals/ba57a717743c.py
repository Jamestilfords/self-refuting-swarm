import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Cosine agreement with the classifier-predicted intent prototype measures angular semantic fit after removing "
    "feature magnitude. Class-conditional empirical calibration flags unusually weak alignment without penalizing intents whose representations are naturally more dispersed."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0140"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = tr.labels
    n_classes = tr.logits.shape[1]

    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    counts = np.bincount(labels, minlength=n_classes).astype(float)
    sums = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(sums, labels, x_train)

    prototypes = sums / np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_sum_for_item = sums[labels] - x_train
    train_count_for_item = counts[labels] - 1.0
    loo_proto = train_sum_for_item / np.maximum(train_count_for_item[:, None], 1.0)

    singleton = train_count_for_item <= 0.0
    if np.any(singleton):
        loo_proto[singleton] = prototypes[labels[singleton]]
    loo_proto = ctx.h.l2(loo_proto)

    train_cosine = np.sum(x_train * loo_proto, axis=1)
    predicted = np.argmax(ev.logits, axis=1)
    eval_cosine = np.sum(x_eval * prototypes[predicted], axis=1)

    tail = np.empty(eval_cosine.shape[0], dtype=float)
    for c in range(n_classes):
        mask_train = labels == c
        mask_eval = predicted == c
        if not np.any(mask_eval):
            continue
        reference = np.sort(train_cosine[mask_train])
        rank = np.searchsorted(reference, eval_cosine[mask_eval], side="right")
        tail[mask_eval] = 1.0 - rank / max(reference.size, 1)

    angular_distance = 0.5 * (1.0 - np.clip(eval_cosine, -1.0, 1.0))
    return tail + 0.01 * angular_distance
