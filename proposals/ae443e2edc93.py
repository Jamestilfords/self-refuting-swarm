import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Training-supported low-energy predictions retain the energy ranking, while items in the "
    "uncertain-energy regime are ranked by classifier-selected pooled-feature support. "
    "Predicted-class robust calibration avoids penalizing intents with naturally broad clusters."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0169", "m_0273", "m_0288"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    cutoff = float(np.quantile(train_energy, 0.70))

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_all_d = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    eval_all_d = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)

    train_d = train_all_d[np.arange(train_all_d.shape[0]), train_pred]
    eval_d = eval_all_d[np.arange(eval_all_d.shape[0]), eval_pred]

    n_classes = tr.logits.shape[1]
    med = np.empty(n_classes, dtype=float)
    scale = np.empty(n_classes, dtype=float)

    global_med = float(np.median(train_d))
    global_scale = float(np.median(np.abs(train_d - global_med))) + 1e-6

    for c in range(n_classes):
        values = train_d[train_pred == c]
        if values.size >= 8:
            center = float(np.median(values))
            spread = float(np.median(np.abs(values - center))) + 1e-6
        else:
            center = global_med
            spread = global_scale
        med[c] = center
        scale[c] = spread

    distance_z = (eval_d - med[eval_pred]) / scale[eval_pred]
    routed = cutoff + 1.0 + np.arctan(distance_z) / np.pi

    return np.where(eval_energy <= cutoff, eval_energy, routed)
