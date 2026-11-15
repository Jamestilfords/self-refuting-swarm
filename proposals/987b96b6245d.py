import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions tightly constrain supported intents; a "
    "logit-posterior-weighted mixture of trailing-subspace class likelihoods flags both "
    "unsupported residuals and disagreement between classifier beliefs and feature support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0017"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(float)
    global_mean = x.mean(axis=0)
    means = np.empty((k, d), dtype=float)
    for c in range(k):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)
        else:
            means[c] = global_mean

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(len(x) - k, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    largest = max(float(eigvals[-1]), 1e-12)
    valid = np.flatnonzero(eigvals > largest * 1e-9)
    if valid.size:
        take = valid[:min(48, valid.size)]
    else:
        take = np.arange(min(16, d))

    basis = eigvecs[:, take]
    global_var = np.maximum(eigvals[take], largest * 1e-9)
    train_proj = x @ basis
    eval_proj = z @ basis
    m = basis.shape[1]

    class_proj_means = np.empty((k, m), dtype=float)
    class_vars = np.empty((k, m), dtype=float)
    shrink = 12.0
    for c in range(k):
        if counts[c] > 0:
            pc = train_proj[y == c]
            mu = pc.mean(axis=0)
            raw_var = ((pc - mu) ** 2).mean(axis=0)
            class_proj_means[c] = mu
            class_vars[c] = (
                counts[c] * raw_var + shrink * global_var
            ) / (counts[c] + shrink)
        else:
            class_proj_means[c] = global_mean @ basis
            class_vars[c] = global_var

    class_vars = np.maximum(class_vars, global_var[None, :] * 1e-6)
    log_like = np.empty((len(z), k), dtype=float)
    constant = m * np.log(2.0 * np.pi)
    for c in range(k):
        delta = eval_proj - class_proj_means[c]
        log_like[:, c] = -0.5 * (
            np.sum((delta * delta) / class_vars[c], axis=1)
            + np.sum(np.log(class_vars[c]))
            + constant
        )

    logits = np.asarray(ev.logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_post = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    joint = log_post + log_like
    joint_max = joint.max(axis=1)
    log_support = joint_max + np.log(
        np.exp(joint - joint_max[:, None]).sum(axis=1)
    )

    return -log_support + 1e-4 * ctx.h.energy(logits)
