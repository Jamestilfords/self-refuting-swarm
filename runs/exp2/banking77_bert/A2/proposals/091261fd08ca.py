import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Confident items retain energy-based ordering, while energy-uncertain items receive increasing weight from predicted-intent representation atypicality. Class-conditional distance calibration avoids over-penalizing naturally broad intents."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0063"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    energy_cdf = ctx.h.normalizer("ecdf", train_energy)
    e_score = energy_cdf(ctx.h.energy(ev.logits))

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_all_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_all_dist = g.class_mahalanobis_sq(ev.pooled)
    train_dist = train_all_dist[np.arange(len(train_pred)), train_pred]
    eval_dist = eval_all_dist[np.arange(len(eval_pred)), eval_pred]

    dist_score = np.empty(len(eval_pred), dtype=float)
    global_sorted = np.sort(train_dist)
    for c in np.unique(eval_pred):
        mask_eval = eval_pred == c
        class_dist = train_dist[train_pred == c]
        reference = np.sort(class_dist) if len(class_dist) >= 8 else global_sorted
        dist_score[mask_eval] = (
            np.searchsorted(reference, eval_dist[mask_eval], side="right")
            / float(len(reference) + 1)
        )

    uncertainty = np.clip((e_score - 0.5) / 0.5, 0.0, 1.0)
    return (1.0 - uncertainty) * e_score + uncertainty * dist_score
