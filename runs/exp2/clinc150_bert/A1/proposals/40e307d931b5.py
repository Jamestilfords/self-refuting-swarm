import numpy as np
from scipy.special import logsumexp

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Known intents have class-specific logit shapes: their winning-versus-runner-up margin, centered logit norm, and tempered evidence tend to occupy a compact region. A query with an atypical combination of confidence scale and competition structure receives a larger conditional shape distance."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    def shape_features(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        norm = np.sqrt(np.sum(centered * centered, axis=1))
        tempered = logsumexp(logits / 2.0, axis=1) * 2.0
        return np.column_stack((margin, norm, tempered))

    x_train = shape_features(tr.logits)
    x_eval = shape_features(ev.logits)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)

    global_mean = np.mean(x_train, axis=0)
    centered_global = x_train - global_mean
    global_cov = centered_global.T @ centered_global / max(len(x_train), 1)
    global_cov += np.eye(3) * (1e-4 + 1e-3 * np.trace(global_cov) / 3.0)

    means = np.empty((k, 3), dtype=float)
    inv_covs = np.empty((k, 3, 3), dtype=float)

    for c in range(k):
        members = x_train[pred_train == c]
        n = len(members)
        shrink = n / (n + 12.0)
        if n:
            local_mean = np.mean(members, axis=0)
        else:
            local_mean = global_mean
        means[c] = shrink * local_mean + (1.0 - shrink) * global_mean

        if n >= 3:
            d = members - local_mean
            local_cov = d.T @ d / n
        else:
            local_cov = global_cov
        cov = shrink * local_cov + (1.0 - shrink) * global_cov
        cov += np.eye(3) * (1e-6 + 1e-4 * np.trace(global_cov) / 3.0)
        inv_covs[c] = np.linalg.pinv(cov)

    delta = x_eval - means[pred_eval]
    score_values = np.einsum("ni,nij,nj->n", delta, inv_covs[pred_eval], delta)
    return np.nan_to_num(score_values, nan=0.0, posinf=1e6, neginf=0.0)
