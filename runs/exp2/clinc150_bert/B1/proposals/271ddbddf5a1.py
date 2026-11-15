import numpy as np
from scipy.special import logsumexp

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A feature component outside the principal in-scope subspace is large for queries that violate dominant encoder variation patterns. ViM converts this residual into a virtual OOD logit and compares it directly against the classifier's in-scope logits."
SOURCE = "https://doi.org/10.48550/arxiv.2203.10807"
PARENTS = ["energy"]
MEMORY_USED = ["m_0046", "m_0047"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    mean = np.mean(tr.pooled, axis=0)
    centered_train = tr.pooled - mean
    cov = centered_train.T @ centered_train / max(len(centered_train), 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    dim = min(256, eigvecs.shape[1])
    basis = eigvecs[:, -dim:]

    train_residual = centered_train - (centered_train @ basis) @ basis.T
    eval_centered = ev.pooled - mean
    eval_residual = eval_centered - (eval_centered @ basis) @ basis.T

    train_norm = np.sqrt(np.sum(train_residual * train_residual, axis=1))
    eval_norm = np.sqrt(np.sum(eval_residual * eval_residual, axis=1))

    mean_residual = max(float(np.mean(train_norm)), 1e-12)
    mean_peak_logit = float(np.mean(np.max(tr.logits, axis=1)))
    alpha = max(mean_peak_logit / mean_residual, 1e-6)

    virtual_logit = alpha * eval_norm
    logit_partition = logsumexp(ev.logits, axis=1)
    return 1.0 / (1.0 + np.exp(-np.clip(virtual_logit - logit_partition, -60.0, 60.0)))
