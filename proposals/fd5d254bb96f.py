import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Confident queries retain their energy score exactly, while queries beyond the "
    "training uncertainty threshold are ranked by how atypical their representation is "
    "for the classifier-selected intent. This hard routing makes feature support decisive "
    "only where energy is already uncertain."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0153", "m_0156"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    # The lower-energy region is retained unchanged.  The threshold is learned only
    # from known in-scope training queries.
    threshold = float(np.quantile(train_energy, 0.65))

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_dist_all = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist_all = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_dist = train_dist_all[np.arange(train_dist_all.shape[0]), train_pred]
    eval_dist = eval_dist_all[np.arange(eval_dist_all.shape[0]), eval_pred]

    # Predicted-intent-specific empirical tail ranks account for intents with
    # intrinsically different representation spread.
    fallback = np.sort(np.nan_to_num(train_dist, nan=0.0, posinf=1e12, neginf=0.0))
    tail = np.empty(eval_dist.shape[0], dtype=float)
    for cls in range(train_dist_all.shape[1]):
        mask_train = train_pred == cls
        ref = train_dist[mask_train]
        if ref.size < 8:
            ref = fallback
        else:
            ref = np.sort(np.nan_to_num(ref, nan=0.0, posinf=1e12, neginf=0.0))

        mask_eval = eval_pred == cls
        if np.any(mask_eval):
            values = np.nan_to_num(
                eval_dist[mask_eval], nan=0.0, posinf=1e12, neginf=0.0
            )
            tail[mask_eval] = np.searchsorted(ref, values, side="right") / float(
                ref.size
            )

    # All uncertain items rank above confident ones, but their ordering is determined
    # by class-conditional feature unsupportedness rather than another energy blend.
    uncertain = eval_energy > threshold
    out = np.array(eval_energy, copy=True)
    out[uncertain] = threshold + 1e-6 + tail[uncertain]

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
