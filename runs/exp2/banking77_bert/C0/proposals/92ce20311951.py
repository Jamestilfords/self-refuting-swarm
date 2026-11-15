import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Class-conditional empirical tail calibration puts energy and feature-space Mahalanobis distance on a shared train-derived percentile scale. Combining their calibrated log-odds emphasizes examples that are simultaneously atypical in classifier confidence and representation space."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    train_class_dist = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    eval_class_dist = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)

    train_dist = train_class_dist[np.arange(len(tr.labels)), tr.labels]
    eval_pred = np.argmax(ev.logits, axis=1)
    eval_dist = eval_class_dist[np.arange(len(eval_pred)), eval_pred]

    n_classes = tr.logits.shape[1]
    energy_p = np.empty(len(ev.logits), dtype=float)
    dist_p = np.empty(len(ev.logits), dtype=float)

    for c in range(n_classes):
        mask = tr.labels == c
        e_ref = np.sort(train_energy[mask])
        d_ref = np.sort(train_dist[mask])
        n = max(len(e_ref), 1)

        use = eval_pred == c
        if np.any(use):
            energy_p[use] = (np.searchsorted(e_ref, eval_energy[use], side="right") + 1.0) / (n + 2.0)
            dist_p[use] = (np.searchsorted(d_ref, eval_dist[use], side="right") + 1.0) / (n + 2.0)

    energy_odds = np.log(energy_p / (1.0 - energy_p))
    dist_odds = np.log(dist_p / (1.0 - dist_p))
    return 0.45 * energy_odds + 0.55 * dist_odds + 0.20 * energy_odds * dist_odds
