import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Classifier uncertainty and bottleneck-space support are complementary signals. "
    "Class-conditional train-fitted upper-tail probabilities make clipped energy and selected-intent Mahalanobis distance comparable before Fisher-style fusion."
)
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0010", "m_0021", "m_0213"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_energy = ctx.h.energy_clip(tr.logits)
    eval_energy = ctx.h.energy_clip(ev.logits)

    train_all_d = g.class_mahalanobis_sq(tr.feats)
    eval_all_d = g.class_mahalanobis_sq(ev.feats)
    train_d = train_all_d[np.arange(len(tr.feats)), train_pred]
    eval_d = eval_all_d[np.arange(len(ev.feats)), eval_pred]

    e_tail = np.empty(len(ev.feats), dtype=float)
    d_tail = np.empty(len(ev.feats), dtype=float)
    classes = np.unique(tr.labels)

    for c in classes:
        mask_tr = train_pred == c
        mask_ev = eval_pred == c

        if np.sum(mask_tr) < 8:
            mask_tr = tr.labels == c

        ref_e = np.sort(train_energy[mask_tr])
        ref_d = np.sort(train_d[mask_tr])
        n = len(ref_e)

        if np.any(mask_ev):
            e_tail[mask_ev] = (
                np.searchsorted(ref_e, eval_energy[mask_ev], side="right") + 1.0
            ) / (n + 2.0)
            d_tail[mask_ev] = (
                np.searchsorted(ref_d, eval_d[mask_ev], side="right") + 1.0
            ) / (n + 2.0)

    e_tail = np.clip(e_tail, 1e-6, 1.0 - 1e-6)
    d_tail = np.clip(d_tail, 1e-6, 1.0 - 1e-6)

    return -2.0 * (np.log1p(-e_tail) + np.log1p(-d_tail))
