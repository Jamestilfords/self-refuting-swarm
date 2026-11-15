import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Energy remains the score for clearly in-scope-looking items, while a train-calibrated "
    "semantic neighborhood radius is smoothly emphasized only in the uncertain energy tail."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0089", "m_0103"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    # The kNN radius supplies local semantic support without fitting an
    # eval-dependent representation model.
    train_radius = np.asarray(ctx.h.knn(tr.pooled, tr.pooled, k=11, l2=True), dtype=float)
    eval_radius = np.asarray(ctx.h.knn(tr.pooled, ev.pooled, k=11, l2=True), dtype=float)

    e_sorted = np.sort(train_energy)
    r_sorted = np.sort(train_radius)
    n = float(max(len(train_energy), 1))

    e_tail = np.searchsorted(e_sorted, eval_energy, side="right") / n
    r_tail = np.searchsorted(r_sorted, eval_radius, side="right") / n

    q50, q75 = np.percentile(train_energy, [50.0, 75.0])
    scale = max(q75 - q50, 1e-6)
    gate = 1.0 / (1.0 + np.exp(-np.clip((eval_energy - q75) / scale, -30.0, 30.0)))

    return e_tail + gate * r_tail
