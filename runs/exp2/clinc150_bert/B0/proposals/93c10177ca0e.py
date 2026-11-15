import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "L2-normalized class prototypes measure whether a query points in the angular direction expected for its classifier-assigned intent. Per-intent empirical lower-tail calibration makes low prototype similarity anomalous relative to the natural angular spread of that intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_tr = np.asarray(tr.feats, dtype=float)
    x_ev = np.asarray(ev.feats, dtype=float)
    y_tr = np.asarray(tr.labels, dtype=int)

    x_tr = x_tr / np.maximum(np.linalg.norm(x_tr, axis=1, keepdims=True), 1e-12)
    x_ev = x_ev / np.maximum(np.linalg.norm(x_ev, axis=1, keepdims=True), 1e-12)

    n_classes = max(
        int(np.max(y_tr)) + 1,
        int(np.asarray(tr.logits).shape[1]),
        int(np.asarray(ev.logits).shape[1]),
    )
    prototypes = np.zeros((n_classes, x_tr.shape[1]), dtype=float)
    counts = np.bincount(y_tr, minlength=n_classes).astype(float)
    np.add.at(prototypes, y_tr, x_tr)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12
    )

    train_cos = np.sum(x_tr * prototypes[y_tr], axis=1)
    global_sorted = np.sort(train_cos)

    sorted_by_class = []
    for c in range(n_classes):
        values = train_cos[y_tr == c]
        sorted_by_class.append(np.sort(values) if values.size else global_sorted)

    pred = np.argmax(np.asarray(ev.logits), axis=1)
    pred = np.clip(pred, 0, n_classes - 1)
    eval_cos = np.sum(x_ev * prototypes[pred], axis=1)

    scores = np.empty(eval_cos.shape[0], dtype=float)
    for c in np.unique(pred):
        mask = pred == c
        reference = sorted_by_class[int(c)]
        rank = np.searchsorted(reference, eval_cos[mask], side="right")
        lower_tail = rank / float(reference.size + 1)
        scores[mask] = 1.0 - lower_tail

    return np.clip(scores, 0.0, 1.0)
