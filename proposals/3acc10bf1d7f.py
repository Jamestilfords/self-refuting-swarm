import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Confident queries retain their original energy ranking, while train-energy-uncertain "
    "queries are ranked by classifier-selected pooled-feature support."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0051", "m_0169", "m_0235"]


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

    train_pred = np.clip(train_pred, 0, train_all_d.shape[1] - 1)
    eval_pred = np.clip(eval_pred, 0, eval_all_d.shape[1] - 1)
    train_d = train_all_d[np.arange(train_all_d.shape[0]), train_pred]
    eval_d = eval_all_d[np.arange(eval_all_d.shape[0]), eval_pred]

    uncertain_train = train_energy > cutoff
    ref = train_d[uncertain_train]
    if ref.size < 8:
        ref = train_d

    d_loc = float(np.quantile(ref, 0.50))
    d_scale = float(np.quantile(ref, 0.90) - d_loc)
    d_scale = max(d_scale, 1e-6)

    support_rank = np.log1p(np.maximum(0.0, (eval_d - d_loc) / d_scale))
    support_rank = np.nan_to_num(support_rank, nan=0.0, posinf=50.0, neginf=0.0)

    confident = eval_energy <= cutoff
    out = np.where(confident, eval_energy, cutoff + support_rank)
    return np.nan_to_num(out, nan=cutoff, posinf=cutoff + 50.0, neginf=cutoff - 50.0)
