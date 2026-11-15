import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Cosine support from the classifier-selected normalized prototype should be high for "
    "in-scope examples, but its typical level varies by predicted intent. Predicted-class "
    "conditional lower-tail calibration highlights angular mismatch without globally ranking "
    "examples solely by their nearest prototype similarity."
)
SOURCE = "https://github.com/pradhyumnaag30/CLINC150"
PARENTS = []
MEMORY_USED = ["m_0075", "m_0114", "m_0122"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    z_tr = ctx.h.l2(tr.pooled)
    z_ev = ctx.h.l2(ev.pooled)
    y = tr.labels.astype(int)

    n_classes = tr.logits.shape[1]
    counts = np.bincount(y, minlength=n_classes).astype(float)
    prototypes = np.zeros((n_classes, z_tr.shape[1]), dtype=float)
    np.add.at(prototypes, y, z_tr)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    train_support = np.sum(z_tr * prototypes[pred_tr], axis=1)
    eval_support = np.sum(z_ev * prototypes[pred_ev], axis=1)

    global_sorted = np.sort(train_support)
    scores = np.empty(eval_support.shape[0], dtype=float)

    for c in range(n_classes):
        mask_train = pred_tr == c
        mask_eval = pred_ev == c
        reference = train_support[mask_train]
        if reference.size < 8:
            reference = global_sorted
        else:
            reference = np.sort(reference)

        rank = np.searchsorted(reference, eval_support[mask_eval], side="right")
        lower_tail = rank / (reference.size + 1.0)
        scores[mask_eval] = 1.0 - lower_tail

    tie_direction = np.linspace(-1.0, 1.0, z_ev.shape[1])
    scores += 1e-8 * (z_ev @ tie_direction)
    return scores
