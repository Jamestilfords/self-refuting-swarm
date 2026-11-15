import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy remains the score for confidently in-scope-like items, while a train-calibrated "
    "predicted-intent Mahalanobis tail score increasingly dominates when energy indicates uncertainty."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0077"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    tr_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    ev_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    med_e = float(np.median(tr_energy))
    q25_e, q75_e = np.percentile(tr_energy, [25.0, 75.0])
    scale_e = max(float(q75_e - q25_e), 1e-6)

    tr_ez = (tr_energy - med_e) / scale_e
    ev_ez = (ev_energy - med_e) / scale_e

    gate_center = float(np.percentile(tr_ez, 75.0))
    gate_scale = max(float(np.percentile(tr_ez, 75.0) - np.percentile(tr_ez, 25.0)), 0.5)
    gate = 1.0 / (1.0 + np.exp(-np.clip((ev_ez - gate_center) / gate_scale, -30.0, 30.0)))

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    tr_all_d = g.class_mahalanobis_sq(tr.pooled)
    ev_all_d = g.class_mahalanobis_sq(ev.pooled)
    tr_d = np.take_along_axis(tr_all_d, tr_pred[:, None], axis=1).ravel()
    ev_d = np.take_along_axis(ev_all_d, ev_pred[:, None], axis=1).ravel()

    global_sorted = np.sort(np.nan_to_num(tr_d, nan=0.0, posinf=1e12, neginf=-1e12))
    tail_prob = np.empty(ev_d.shape[0], dtype=float)

    for c in range(tr_all_d.shape[1]):
        train_for_class = tr_d[tr_pred == c]
        reference = np.sort(train_for_class) if train_for_class.size >= 8 else global_sorted
        mask = ev_pred == c
        if np.any(mask):
            ranks = np.searchsorted(reference, ev_d[mask], side="right")
            tail_prob[mask] = (ranks + 0.5) / (reference.size + 1.0)

    tail_prob = np.clip(tail_prob, 1e-4, 1.0 - 1e-4)
    distance_score = np.log(tail_prob / (1.0 - tail_prob))

    scores = (1.0 - gate) * ev_ez + gate * distance_score
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
