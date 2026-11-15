import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "OOS queries may be assigned a confident class while remaining poorly aligned with that predicted "
    "intent's normalized feature prototype. Class-specific training calibration accounts for intents "
    "whose in-scope examples naturally have different prototype-consistency levels."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0014", "m_0031"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)

    global_proto = np.mean(x_train, axis=0, keepdims=True)
    global_proto = ctx.h.l2(global_proto)[0]
    nonempty = counts > 0
    prototypes[nonempty] /= counts[nonempty, None]
    prototypes[~nonempty] = global_proto
    prototypes = ctx.h.l2(prototypes)

    train_similarity = np.sum(x_train * prototypes[labels], axis=1)
    sim_sum = np.bincount(labels, weights=train_similarity, minlength=n_classes)
    sim_mean = np.zeros(n_classes, dtype=float)
    sim_mean[nonempty] = sim_sum[nonempty] / counts[nonempty]
    sim_mean[~nonempty] = np.mean(train_similarity)

    centered_sq = (train_similarity - sim_mean[labels]) ** 2
    sim_var = np.bincount(labels, weights=centered_sq, minlength=n_classes)
    sim_std = np.sqrt(sim_var / np.maximum(counts, 1.0))
    fallback_std = max(float(np.std(train_similarity)), 1e-4)
    sim_std = np.maximum(sim_std, fallback_std * 0.1)
    sim_std[~nonempty] = fallback_std

    predicted = np.argmax(ev.logits, axis=1)
    eval_similarity = np.sum(x_eval * prototypes[predicted], axis=1)
    return (sim_mean[predicted] - eval_similarity) / sim_std[predicted]
