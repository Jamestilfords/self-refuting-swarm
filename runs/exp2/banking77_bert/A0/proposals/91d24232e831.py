import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Classifier-predicted-intent calibration makes logit uncertainty and predicted-class "
    "representation mismatch comparable on train-fitted percentile scales. OOS inputs should "
    "simultaneously have unusually weak logit evidence or atypical encoder-space distance."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0073", "m_0077"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    train_dist_all = g.class_mahalanobis_sq(tr.pooled)
    eval_dist_all = g.class_mahalanobis_sq(ev.pooled)
    train_dist = train_dist_all[np.arange(len(tr.logits)), train_pred]
    eval_dist = eval_dist_all[np.arange(len(ev.logits)), eval_pred]

    energy_q = np.empty(len(ev.logits), dtype=float)
    dist_q = np.empty(len(ev.logits), dtype=float)

    for c in range(tr.logits.shape[1]):
        mask = train_pred == c
        if np.any(mask):
            e_ref = np.sort(train_energy[mask])
            d_ref = np.sort(train_dist[mask])
        else:
            e_ref = np.sort(train_energy)
            d_ref = np.sort(train_dist)

        idx = eval_pred == c
        energy_q[idx] = np.searchsorted(e_ref, eval_energy[idx], side="right") / len(e_ref)
        dist_q[idx] = np.searchsorted(d_ref, eval_dist[idx], side="right") / len(d_ref)

    return 0.5 * energy_q + 0.5 * dist_q
