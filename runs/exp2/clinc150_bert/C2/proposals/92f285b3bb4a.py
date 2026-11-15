import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "OOS inputs can have unusual Dirichlet concentration geometry even when their total evidence is similar to in-scope examples. This score measures class-conditional deviation in total concentration, normalized concentration entropy, and concentration peakiness."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=float)
        alpha = np.clip(alpha, 1e-8, 1e12)
        total = np.sum(alpha, axis=1)
        probs = alpha / total[:, None]
        k = alpha.shape[1]
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        entropy = entropy / max(np.log(max(k, 2)), 1e-8)
        peak = np.max(alpha, axis=1) / total
        return np.column_stack((
            np.log(total),
            entropy,
            np.log(np.clip(peak, 1e-8, 1.0))
        ))

    x_train = shape_features(tr.logits)
    x_eval = shape_features(ev.logits)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    global_mean = np.mean(x_train, axis=0)
    centered = x_train - global_mean
    denom = max(x_train.shape[0] - 1, 1)
    cov = (centered.T @ centered) / denom
    scale = np.trace(cov) / cov.shape[0]
    cov = cov + np.eye(cov.shape[0]) * max(scale * 1e-4, 1e-6)
    precision = np.linalg.pinv(cov)

    means = np.tile(global_mean, (n_classes, 1))
    for c in range(n_classes):
        members = x_train[labels == c]
        if members.shape[0] > 0:
            means[c] = np.mean(members, axis=0)

    pred = np.argmax(ev.logits, axis=1)
    residual = x_eval - means[pred]
    score_values = np.einsum("nd,dd,nd->n", residual, precision, residual)

    total_evidence = x_eval[:, 0]
    score_values = score_values + 1e-3 * (total_evidence - global_mean[0]) ** 2
    return np.nan_to_num(score_values, nan=0.0, posinf=1e12, neginf=0.0)
