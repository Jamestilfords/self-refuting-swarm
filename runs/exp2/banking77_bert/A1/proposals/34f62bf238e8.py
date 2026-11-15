import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy is retained for confident predictions, while a predicted-intent bottleneck "
    "Mahalanobis distance increasingly determines the score when energy indicates uncertainty. "
    "This can expose uncertain queries that are not locally compatible with their assigned intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0019", "m_0025"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    train_all_dist = np.asarray(g.class_mahalanobis_sq(tr.feats), dtype=float)
    eval_all_dist = np.asarray(g.class_mahalanobis_sq(ev.feats), dtype=float)

    train_labels = np.asarray(tr.labels, dtype=int)
    train_dist = train_all_dist[np.arange(train_labels.shape[0]), train_labels]

    eval_pred = np.argmax(ev.logits, axis=1)
    eval_dist = eval_all_dist[np.arange(eval_pred.shape[0]), eval_pred]

    e_ref = np.sort(np.nan_to_num(train_energy, nan=0.0, posinf=1e12, neginf=-1e12))
    d_ref = np.sort(np.nan_to_num(train_dist, nan=0.0, posinf=1e12, neginf=-1e12))

    def ecdf(x, reference):
        x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=1e12, neginf=-1e12)
        return np.searchsorted(reference, x, side="right") / float(reference.size + 1)

    energy_score = ecdf(eval_energy, e_ref)
    distance_score = ecdf(eval_dist, d_ref)

    train_energy_q = ecdf(train_energy, e_ref)
    center = float(np.quantile(train_energy_q, 0.60))
    spread = float(np.quantile(train_energy_q, 0.85) - np.quantile(train_energy_q, 0.35))
    spread = max(spread, 0.08)

    gate = 1.0 / (1.0 + np.exp(-6.0 * (energy_score - center) / spread))
    scores = (1.0 - gate) * energy_score + gate * distance_score
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
