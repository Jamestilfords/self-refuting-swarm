import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Class-conditional empirical tail calibration makes logit uncertainty and distance-to-the-predicted-intent comparable despite different intent-specific score ranges. Fisher-style aggregation highlights queries that are unusually uncertain, geometrically implausible, or both."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0045"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    train_d_all = g.class_mahalanobis_sq(tr.pooled)
    eval_d_all = g.class_mahalanobis_sq(ev.pooled)
    train_d = train_d_all[np.arange(train_d_all.shape[0]), train_pred]
    eval_d = eval_d_all[np.arange(eval_d_all.shape[0]), eval_pred]

    n_eval = eval_energy.shape[0]
    q_energy = np.empty(n_eval, dtype=float)
    q_dist = np.empty(n_eval, dtype=float)

    global_e = np.sort(train_energy)
    global_d = np.sort(train_d)
    n_global = float(train_energy.shape[0])

    for c in range(tr.logits.shape[1]):
        mask_train = train_pred == c
        mask_eval = eval_pred == c
        if not np.any(mask_eval):
            continue

        if np.sum(mask_train) >= 8:
            e_ref = np.sort(train_energy[mask_train])
            d_ref = np.sort(train_d[mask_train])
        else:
            e_ref = global_e
            d_ref = global_d

        denom = float(e_ref.shape[0] + 1)
        q_energy[mask_eval] = (
            np.searchsorted(e_ref, eval_energy[mask_eval], side="right") / denom
        )
        q_dist[mask_eval] = (
            np.searchsorted(d_ref, eval_d[mask_eval], side="right") / denom
        )

    q_energy = np.clip(q_energy, 0.0, 1.0 - 1e-12)
    q_dist = np.clip(q_dist, 0.0, 1.0 - 1e-12)
    return -2.0 * (np.log1p(-q_energy) + np.log1p(-q_dist))
