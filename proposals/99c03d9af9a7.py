import numpy as np
from scipy.special import logsumexp

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "A generalized winner-versus-rivals logit score captures uncertainty spread across all competing intents, "
    "while predicted-intent Mahalanobis surprise measures whether the representation supports that decision. "
    "Class-conditional train-fitted tail maps put the complementary cues on a common scale."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0021", "m_0268", "m_0273"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    def rival_mass(logits):
        winner = np.argmax(logits, axis=1)
        top = logits[np.arange(logits.shape[0]), winner]
        masked = logits.copy()
        masked[np.arange(logits.shape[0]), winner] = -np.inf
        return logsumexp(masked - top[:, None], axis=1)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_logit = rival_mass(tr.logits)
    eval_logit = rival_mass(ev.logits)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist_all = g.class_mahalanobis_sq(tr.pooled)
    eval_dist_all = g.class_mahalanobis_sq(ev.pooled)
    train_dist = train_dist_all[np.arange(len(tr.pooled)), train_pred]
    eval_dist = eval_dist_all[np.arange(len(ev.pooled)), eval_pred]

    train_fused = np.empty(len(tr.pooled), dtype=float)
    eval_fused = np.empty(len(ev.pooled), dtype=float)

    for c in range(k):
        idx = np.flatnonzero(train_pred == c)
        eidx = np.flatnonzero(eval_pred == c)

        if idx.size < 8:
            idx = np.arange(len(tr.pooled))

        a = np.sort(train_logit[idx])
        b = np.sort(train_dist[idx])

        train_a_tail = np.searchsorted(a, train_logit, side="right") / (a.size + 1.0)
        train_b_tail = np.searchsorted(b, train_dist, side="right") / (b.size + 1.0)
        eval_a_tail = np.searchsorted(a, eval_logit, side="right") / (a.size + 1.0)
        eval_b_tail = np.searchsorted(b, eval_dist, side="right") / (b.size + 1.0)

        train_mask = train_pred == c
        eval_mask = eval_pred == c
        train_fused[train_mask] = 0.5 * (
            train_a_tail[train_mask] + train_b_tail[train_mask]
        )
        eval_fused[eval_mask] = 0.5 * (
            eval_a_tail[eval_mask] + eval_b_tail[eval_mask]
        )

    return np.nan_to_num(eval_fused, nan=0.5, posinf=1.0, neginf=0.0)
