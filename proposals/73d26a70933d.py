import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "A query is suspicious when its top-two margin is unusually weak for its overall logit scale "
    "and its normalized pooled representation lacks class-conditional Mahalanobis support. "
    "Train-fitted class-conditional empirical tail maps put these complementary anomalies on a common scale."
)
SOURCE = "https://github.com/mueller-mp/maha-norm"
PARENTS = []
MEMORY_USED = ["m_0081", "m_0088"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    k = tr.logits.shape[1]

    def logit_features(z):
        centered = z - np.mean(z, axis=1, keepdims=True)
        scale = np.sqrt(np.mean(centered * centered, axis=1) + eps)
        top2 = np.partition(z, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        return scale, margin

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)
    tr_scale, tr_margin = logit_features(tr.logits)
    ev_scale, ev_margin = logit_features(ev.logits)

    global_x = np.column_stack([np.ones(len(tr_scale)), tr_scale])
    global_coef = np.linalg.solve(
        global_x.T @ global_x + 1e-4 * np.eye(2),
        global_x.T @ tr_margin,
    )

    tr_logit = np.empty(len(tr_scale), dtype=float)
    ev_logit = np.empty(len(ev_scale), dtype=float)
    for c in range(k):
        mask = tr_pred == c
        if np.sum(mask) >= 8:
            x = np.column_stack([np.ones(np.sum(mask)), tr_scale[mask]])
            coef = np.linalg.solve(
                x.T @ x + 1e-4 * np.eye(2),
                x.T @ tr_margin[mask],
            )
        else:
            coef = global_coef

        tr_logit[mask] = coef[0] + coef[1] * tr_scale[mask] - tr_margin[mask]
        emask = ev_pred == c
        ev_logit[emask] = coef[0] + coef[1] * ev_scale[emask] - ev_margin[emask]

    tr_unit = ctx.h.l2(tr.pooled)
    ev_unit = ctx.h.l2(ev.pooled)
    g = ctx.h.gda(tr_unit, tr.labels, cov="tied")
    tr_all_d = g.class_mahalanobis_sq(tr_unit)
    ev_all_d = g.class_mahalanobis_sq(ev_unit)
    tr_dist = tr_all_d[np.arange(len(tr_pred)), tr_pred]
    ev_dist = ev_all_d[np.arange(len(ev_pred)), ev_pred]

    def class_tail(train_values, eval_values):
        out = np.empty(len(eval_values), dtype=float)
        fallback = np.sort(np.nan_to_num(train_values, nan=0.0, posinf=1e12, neginf=-1e12))
        for c in range(k):
            mask = tr_pred == c
            ref = train_values[mask] if np.sum(mask) >= 8 else fallback
            ref = np.sort(np.nan_to_num(ref, nan=0.0, posinf=1e12, neginf=-1e12))
            emask = ev_pred == c
            vals = np.nan_to_num(eval_values[emask], nan=0.0, posinf=1e12, neginf=-1e12)
            out[emask] = np.searchsorted(ref, vals, side="right") / (len(ref) + 1.0)
        return out

    logit_tail = class_tail(tr_logit, ev_logit)
    dist_tail = class_tail(tr_dist, ev_dist)

    score_out = 1.0 - (1.0 - logit_tail) * (1.0 - dist_tail)
    return np.nan_to_num(score_out, nan=0.0, posinf=1.0, neginf=0.0)
