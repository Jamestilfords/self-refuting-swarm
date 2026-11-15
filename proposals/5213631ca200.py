import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "Beyond confidence, in-scope predictions have class-specific structure in the winning "
    "margin and in how probability mass is distributed among losing intents. A query with an "
    "atypical competitor-tail shape is suspicious even when its maximum probability is high."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0008", "m_0113"]


def score(ctx):
    def shape_features(logits):
        z = np.asarray(logits, dtype=float)
        z = np.nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0)
        n, k = z.shape

        order = np.partition(z, kth=k - 2, axis=1)
        top = order[:, -1]
        second = order[:, -2]
        margin = np.log1p(np.maximum(top - second, 0.0))

        winner = np.argmax(z, axis=1)
        masked = z.copy()
        masked[np.arange(n), winner] = -np.inf
        tail_max = np.max(masked, axis=1, keepdims=True)
        tail_exp = np.exp(masked - tail_max)
        tail_exp[np.arange(n), winner] = 0.0
        tail_p = tail_exp / np.maximum(tail_exp.sum(axis=1, keepdims=True), 1e-12)
        tail_entropy = -np.sum(
            np.where(tail_p > 0.0, tail_p * np.log(np.maximum(tail_p, 1e-12)), 0.0),
            axis=1,
        )
        tail_entropy /= np.log(max(k - 1, 2))

        total = z.sum(axis=1)
        total_sq = (z * z).sum(axis=1)
        tail_mean = (total - top) / max(k - 1, 1)
        tail_var = (total_sq - top * top) / max(k - 1, 1) - tail_mean * tail_mean
        tail_spread = np.log1p(np.sqrt(np.maximum(tail_var, 0.0)))

        return np.column_stack((margin, tail_entropy, tail_spread)), winner

    tr_x, tr_pred = shape_features(ctx.train.logits)
    ev_x, ev_pred = shape_features(ctx.eval.logits)

    d = tr_x.shape[1]
    global_mean = tr_x.mean(axis=0)
    centered = tr_x - global_mean
    global_cov = centered.T @ centered / max(len(tr_x) - 1, 1)
    scale = max(float(np.trace(global_cov)) / d, 1e-6)
    global_cov = global_cov + np.eye(d) * (1e-4 * scale)

    scores = np.empty(len(ev_x), dtype=float)
    for c in np.unique(ev_pred):
        ev_mask = ev_pred == c
        tr_mask = tr_pred == c
        xc = tr_x[tr_mask]
        n_c = len(xc)

        if n_c == 0:
            mean = global_mean
            cov = global_cov
        else:
            raw_mean = xc.mean(axis=0)
            shrink = n_c / (n_c + 12.0)
            mean = shrink * raw_mean + (1.0 - shrink) * global_mean
            local_centered = xc - raw_mean
            local_cov = local_centered.T @ local_centered / max(n_c - 1, 1)
            cov = shrink * local_cov + (1.0 - shrink) * global_cov
            cov = cov + np.eye(d) * (1e-4 * scale)

        inv_cov = np.linalg.pinv(cov)
        delta = ev_x[ev_mask] - mean
        scores[ev_mask] = np.einsum("ij,jk,ik->i", delta, inv_cov, delta)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
