import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The temperature-indexed soft rival mass measures how many competing intents remain plausible "
    "as logits are sharpened or softened. OOS queries can have an atypical multi-temperature rival "
    "spectrum even when their ordinary energy or maximum probability appears in-distribution."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    temperatures = np.asarray([0.35, 0.70, 1.40, 2.80], dtype=float)

    def rival_spectrum(logits):
        z = np.asarray(logits, dtype=float)
        winner = np.max(z, axis=1, keepdims=True)
        gaps = z - winner
        parts = []
        for temp in temperatures:
            a = gaps / temp
            m = np.max(a, axis=1, keepdims=True)
            parts.append((temp * (m[:, 0] + np.log(np.sum(np.exp(a - m), axis=1)))))
        return np.column_stack(parts)

    x_train = rival_spectrum(tr.logits)
    x_eval = rival_spectrum(ev.logits)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)

    dim = x_train.shape[1]
    global_mean = np.mean(x_train, axis=0)
    centered = x_train - global_mean
    global_cov = centered.T @ centered / max(len(x_train) - 1, 1)
    global_scale = float(np.trace(global_cov) / dim)
    global_cov = global_cov + (1e-5 + 1e-3 * max(global_scale, 1e-8)) * np.eye(dim)
    global_inv = np.linalg.pinv(global_cov)

    means = []
    inverses = []
    n_classes = tr.logits.shape[1]
    for c in range(n_classes):
        xc = x_train[pred_train == c]
        if len(xc) < dim + 2:
            means.append(global_mean)
            inverses.append(global_inv)
            continue
        mu = np.mean(xc, axis=0)
        dc = xc - mu
        cov = dc.T @ dc / max(len(xc) - 1, 1)
        scale = float(np.trace(cov) / dim)
        cov = cov + (1e-5 + 2e-3 * max(scale, 1e-8)) * np.eye(dim)
        means.append(mu)
        inverses.append(np.linalg.pinv(cov))

    out = np.empty(len(x_eval), dtype=float)
    for c in range(n_classes):
        mask = pred_eval == c
        if np.any(mask):
            delta = x_eval[mask] - means[c]
            out[mask] = np.einsum("ij,jk,ik->i", delta, inverses[c], delta)

    # A negligible itemwise tie-breaker preserves deterministic ordering in exact spectrum ties.
    out = out + 1e-10 * np.sum(np.square(ev.logits), axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
