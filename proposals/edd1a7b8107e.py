import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Classifier uncertainty and predicted-intent feature support are complementary signals. "
    "Class-conditional training tail probabilities put energy and bottleneck Mahalanobis residuals on a common scale before fusion."
)
SOURCE = "https://github.com/pradhyumnaag30/CLINC150"
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0122", "m_0125"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    train_all_dist = np.asarray(g.class_mahalanobis_sq(tr.feats), dtype=float)
    eval_all_dist = np.asarray(g.class_mahalanobis_sq(ev.feats), dtype=float)
    train_dist = train_all_dist[np.arange(train_all_dist.shape[0]), train_pred]
    eval_dist = eval_all_dist[np.arange(eval_all_dist.shape[0]), eval_pred]

    train_energy = np.nan_to_num(train_energy, nan=0.0, posinf=1e12, neginf=-1e12)
    eval_energy = np.nan_to_num(eval_energy, nan=0.0, posinf=1e12, neginf=-1e12)
    train_dist = np.nan_to_num(train_dist, nan=0.0, posinf=1e12, neginf=-1e12)
    eval_dist = np.nan_to_num(eval_dist, nan=0.0, posinf=1e12, neginf=-1e12)

    global_energy = np.sort(train_energy)
    global_dist = np.sort(train_dist)
    out = np.empty(eval_energy.shape[0], dtype=float)

    for c in range(tr.logits.shape[1]):
        train_mask = train_pred == c
        eval_mask = eval_pred == c
        if not np.any(eval_mask):
            continue

        if np.sum(train_mask) >= 8:
            e_ref = np.sort(train_energy[train_mask])
            d_ref = np.sort(train_dist[train_mask])
        else:
            e_ref = global_energy
            d_ref = global_dist

        e_tail = (e_ref.size - np.searchsorted(e_ref, eval_energy[eval_mask], side="left") + 1.0) / (e_ref.size + 1.0)
        d_tail = (d_ref.size - np.searchsorted(d_ref, eval_dist[eval_mask], side="left") + 1.0) / (d_ref.size + 1.0)

        out[eval_mask] = -np.log(e_tail) - np.log(d_tail)

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
