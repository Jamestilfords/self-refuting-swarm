import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "The fraction of centered encoder-feature energy outside a compact in-scope principal subspace measures representation novelty while discounting overall feature magnitude. This angular residual differs from an energy-dominated ViM fusion and emphasizes directions unsupported by training variation."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    mean = np.mean(x_train, axis=0)
    train_centered = x_train - mean
    eval_centered = x_eval - mean

    cov = (train_centered.T @ train_centered) / max(x_train.shape[0] - 1, 1)
    _, vectors = np.linalg.eigh(cov)

    rank = min(64, vectors.shape[1])
    basis = vectors[:, -rank:]

    eval_total_sq = np.sum(eval_centered * eval_centered, axis=1)
    eval_projected = eval_centered @ basis
    eval_explained_sq = np.sum(eval_projected * eval_projected, axis=1)
    eval_residual_sq = np.maximum(eval_total_sq - eval_explained_sq, 0.0)

    residual_fraction = eval_residual_sq / (eval_total_sq + 1e-12)
    return np.clip(residual_fraction, 0.0, 1.0)
