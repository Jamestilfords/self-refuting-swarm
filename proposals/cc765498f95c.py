import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "In-scope representations tend to fall decisively inside an intent prototype's "
    "Voronoi region, whereas OOS queries often lie near boundaries between competing intents. "
    "Calibrating prototype-boundary ambiguity separately for the classifier-predicted intent "
    "accounts for intents with intrinsically crowded semantic neighborhoods."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0242", "m_0032", "m_0176"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]

    # Small nearest-versus-runner-up gaps indicate prototype-boundary ambiguity.
    train_raw = -(train_two[:, 1] - train_two[:, 0])
    eval_raw = -(eval_two[:, 1] - eval_two[:, 0])

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    out = np.empty(eval_raw.shape[0], dtype=float)

    global_sorted = np.sort(train_raw)
    for c in np.unique(eval_pred):
        mask_train = train_pred == c
        ref = train_raw[mask_train]
        if ref.size < 8:
            ref = global_sorted
        else:
            ref = np.sort(ref)

        mask_eval = eval_pred == c
        ranks = np.searchsorted(ref, eval_raw[mask_eval], side="right")
        out[mask_eval] = ranks / (ref.size + 1.0)

    return out
