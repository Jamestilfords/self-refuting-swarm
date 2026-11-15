import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query is suspicious when its classifier-predicted intent disagrees with the closest Gaussian intent in representation space. Calibrating this predicted-versus-nearest distance disagreement within each predicted intent flags geometrically unsupported assignments without reducing to nearest-class density."
SOURCE = "https://arxiv.org/pdf/2106.09022v1.pdf"
PARENTS = []
MEMORY_USED = ["m_0117", "m_0126"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = train_d.shape[1]

    train_selected = train_d[np.arange(train_d.shape[0]), train_pred]
    eval_selected = eval_d[np.arange(eval_d.shape[0]), eval_pred]

    train_nearest = np.min(train_d, axis=1)
    eval_nearest = np.min(eval_d, axis=1)

    train_disagreement = train_selected - train_nearest
    eval_disagreement = eval_selected - eval_nearest

    sorted_refs = []
    for c in range(n_classes):
        ref = train_disagreement[train_pred == c]
        if ref.size < 8:
            ref = train_disagreement[tr.labels == c]
        if ref.size == 0:
            ref = train_disagreement
        sorted_refs.append(np.sort(np.nan_to_num(ref, nan=0.0, posinf=1e6, neginf=-1e6)))

    scores = np.empty(eval_disagreement.shape[0], dtype=float)
    clean_eval = np.nan_to_num(eval_disagreement, nan=0.0, posinf=1e6, neginf=-1e6)

    for c in range(n_classes):
        mask = eval_pred == c
        if not np.any(mask):
            continue
        ref = sorted_refs[c]
        ranks = np.searchsorted(ref, clean_eval[mask], side="right")
        scores[mask] = ranks / (ref.size + 1.0)

    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
