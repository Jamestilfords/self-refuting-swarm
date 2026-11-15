import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope intents have characteristic changes in predictive entropy as the logits are tempered from sharp to soft distributions. "
    "Class-conditional surprise in this temperature-response curve detects unusual winner-versus-rival competition beyond energy or maximum probability."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0255"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    temperatures = (0.45, 1.8)

    def entropy_curve(logits):
        features = []
        for temp in temperatures:
            z = logits / temp
            z = z - np.max(z, axis=1, keepdims=True)
            p = np.exp(z)
            p /= np.maximum(np.sum(p, axis=1, keepdims=True), eps)
            features.append(-np.sum(p * np.log(np.maximum(p, eps)), axis=1))
        return np.column_stack(features)

    train_x = entropy_curve(tr.logits)
    eval_x = entropy_curve(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_mean = np.mean(train_x, axis=0)
    centered = train_x - global_mean
    global_cov = centered.T @ centered / max(train_x.shape[0] - 1, 1)
    global_scale = max(float(np.trace(global_cov)) / 2.0, 1e-5)
    global_cov = global_cov + np.eye(2) * (1e-4 * global_scale)
    global_inv = np.linalg.pinv(global_cov)

    means = np.empty((n_classes, 2), dtype=float)
    inv_covs = np.empty((n_classes, 2, 2), dtype=float)

    for c in range(n_classes):
        members = train_x[train_pred == c]
        count = members.shape[0]
        if count < 4:
            means[c] = global_mean
            inv_covs[c] = global_inv
            continue

        mean = np.mean(members, axis=0)
        residual = members - mean
        cov = residual.T @ residual / max(count - 1, 1)
        shrink = 12.0 / (count + 12.0)
        cov = (1.0 - shrink) * cov + shrink * global_cov
        cov += np.eye(2) * (1e-5 * global_scale)

        means[c] = mean
        inv_covs[c] = np.linalg.pinv(cov)

    delta = eval_x - means[eval_pred]
    score_values = np.einsum(
        "ni,nij,nj->n", delta, inv_covs[eval_pred], delta, optimize=True
    )
    return np.nan_to_num(score_values, nan=0.0, posinf=1e6, neginf=0.0)
