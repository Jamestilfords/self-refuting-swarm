import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A supported prediction should agree with feature geometry: its predicted intent "
    "should be nearly as close as the best class prototype. Class-conditional tail "
    "calibration flags classifier-to-representation disagreement without relying on eval statistics."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0032", "m_0112"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")

    train_d = np.nan_to_num(
        g.class_mahalanobis_sq(tr.feats), nan=0.0, posinf=1e12, neginf=0.0
    )
    eval_d = np.nan_to_num(
        g.class_mahalanobis_sq(ev.feats), nan=0.0, posinf=1e12, neginf=0.0
    )

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_idx = np.arange(train_d.shape[0])
    eval_idx = np.arange(eval_d.shape[0])

    train_selected = train_d[train_idx, train_pred]
    eval_selected = eval_d[eval_idx, eval_pred]
    train_gap = np.maximum(train_selected - np.min(train_d, axis=1), 0.0)
    eval_gap = np.maximum(eval_selected - np.min(eval_d, axis=1), 0.0)

    global_sorted = np.sort(train_gap)
    scores = np.empty(eval_gap.shape[0], dtype=float)

    for c in range(train_d.shape[1]):
        mask_train = train_pred == c
        mask_eval = eval_pred == c
        reference = np.sort(train_gap[mask_train])
        if reference.size < 8:
            reference = global_sorted
        ranks = np.searchsorted(reference, eval_gap[mask_eval], side="right")
        tail = ranks / float(reference.size + 1)
        scale = np.median(reference) + 1e-6
        scores[mask_eval] = tail + 1e-4 * eval_gap[mask_eval] / scale

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
