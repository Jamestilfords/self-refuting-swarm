import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy is retained exactly for confident queries, but uncertain queries are reranked primarily "
    "by class-conditional pooled-feature support. This prevents energy from dominating the uncertain "
    "regime when a query is geometrically atypical of the intent selected by the classifier."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0051", "m_0169", "m_0200"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    threshold = float(np.quantile(train_energy, 0.60))
    q25, q75 = np.quantile(train_energy, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-6)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_dist_all = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist_all = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)
    train_dist = train_dist_all[np.arange(len(train_pred)), train_pred]
    eval_dist = eval_dist_all[np.arange(len(eval_pred)), eval_pred]

    train_dist = np.nan_to_num(train_dist, nan=0.0, posinf=1e12, neginf=0.0)
    eval_dist = np.nan_to_num(eval_dist, nan=0.0, posinf=1e12, neginf=0.0)

    global_ref = np.sort(train_dist)
    refs = []
    for c in range(tr.logits.shape[1]):
        values = train_dist[train_pred == c]
        if values.size >= 12:
            refs.append(np.sort(values))
        else:
            refs.append(global_ref)

    tail = np.empty(len(eval_dist), dtype=float)
    for c in range(ev.logits.shape[1]):
        mask = eval_pred == c
        if np.any(mask):
            ref = refs[c]
            tail[mask] = np.searchsorted(
                ref, eval_dist[mask], side="right"
            ) / float(ref.size + 1)

    confident = eval_energy <= threshold
    out = np.empty(len(eval_energy), dtype=float)

    # Preserve the original energy value and ranking in the confident regime.
    out[confident] = eval_energy[confident]

    # Above the train-fitted uncertainty boundary, replace the mostly redundant
    # energy ordering with representation-support ranking. A small energy term
    # only resolves ties among equally supported examples.
    uncertain = ~confident
    excess = np.maximum(eval_energy[uncertain] - threshold, 0.0) / scale
    out[uncertain] = (
        threshold
        + scale * (1.0 + 4.0 * tail[uncertain] + 0.03 * excess)
    )

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
