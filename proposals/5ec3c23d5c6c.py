import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual magnitude measures encoder components unsupported by the principal in-scope feature subspace, while energy retains classifier uncertainty information. Train-fitted robust scaling combines these complementary signals without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0004"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    mean = np.mean(x_train, axis=0)
    centered = x_train - mean

    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]

    dim = min(256, x_train.shape[1], max(1, x_train.shape[0] - 1))
    basis = vecs[:, order[:dim]]

    train_residual = centered - (centered @ basis) @ basis.T
    eval_centered = x_eval - mean
    eval_residual = eval_centered - (eval_centered @ basis) @ basis.T

    train_r = np.sqrt(np.maximum(np.sum(train_residual * train_residual, axis=1), 0.0))
    eval_r = np.sqrt(np.maximum(np.sum(eval_residual * eval_residual, axis=1), 0.0))

    train_e = np.asarray(ctx.h.energy(tr.logits), dtype=np.float64)
    eval_e = np.asarray(ctx.h.energy(ev.logits), dtype=np.float64)

    r_med = np.median(train_r)
    e_med = np.median(train_e)
    r_scale = np.median(np.abs(train_r - r_med)) * 1.4826
    e_scale = np.median(np.abs(train_e - e_med)) * 1.4826

    r_scale = max(float(r_scale), 1e-8)
    e_scale = max(float(e_scale), 1e-8)

    residual_score = (eval_r - r_med) / r_scale
    energy_score = (eval_e - e_med) / e_scale
    return energy_score + residual_score
