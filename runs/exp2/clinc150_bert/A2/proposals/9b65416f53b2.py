import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A global principal subspace captures dominant in-scope encoder variation, while the residual norm measures representation energy unsupported by that manifold. Adding a train-scaled residual correction to energy implements a ViM-style novelty signal."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0023"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    mean = x_train.mean(axis=0, keepdims=True)

    centered_train = x_train - mean
    centered_eval = x_eval - mean

    cov = centered_train.T @ centered_train
    cov /= max(len(centered_train) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    rank = min(128, eigvecs.shape[1])
    basis = eigvecs[:, -rank:]

    train_total = np.sum(centered_train * centered_train, axis=1)
    train_proj = np.sum((centered_train @ basis) ** 2, axis=1)
    train_residual = np.sqrt(np.maximum(train_total - train_proj, 0.0))

    eval_total = np.sum(centered_eval * centered_eval, axis=1)
    eval_proj = np.sum((centered_eval @ basis) ** 2, axis=1)
    eval_residual = np.sqrt(np.maximum(eval_total - eval_proj, 0.0))

    train_energy = ctx.h.energy(tr.logits)
    energy_scale = np.std(train_energy)
    residual_scale = np.std(train_residual)
    alpha = energy_scale / max(residual_scale, 1e-12)

    return ctx.h.energy(ev.logits) + alpha * eval_residual
