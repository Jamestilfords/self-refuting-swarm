import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A prediction-conditioned diagonal Mahalanobis score retains intent-specific feature variances while shrinkage stabilizes rare intents. Calibrating each predicted intent's distance against its own in-scope distance distribution exposes queries that the classifier assigns confidently but whose representation is atypical for that intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009", "m_0021"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    pred = np.argmax(ev.logits, axis=1)

    n, d = x.shape
    k = tr.logits.shape[1]
    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0) + 1e-6
    scores = np.empty(z.shape[0], dtype=float)

    for c in range(k):
        mask = y == c
        nc = int(np.sum(mask))
        if nc:
            xc = x[mask]
            mean_c = np.mean(xc, axis=0)
            var_c = np.var(xc, axis=0)
            shrink = 20.0 / (nc + 20.0)
            var_c = (1.0 - shrink) * var_c + shrink * global_var
            var_c = np.maximum(var_c, 1e-6)
            train_d = np.sum((xc - mean_c) ** 2 / var_c, axis=1)
        else:
            mean_c = global_mean
            var_c = global_var
            train_d = np.sum((x - mean_c) ** 2 / var_c, axis=1)

        med = np.median(train_d)
        q25, q75 = np.quantile(train_d, [0.25, 0.75])
        scale = max(q75 - q25, 1e-6)

        idx = pred == c
        if np.any(idx):
            eval_d = np.sum((z[idx] - mean_c) ** 2 / var_c, axis=1)
            scores[idx] = (eval_d - med) / scale

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
