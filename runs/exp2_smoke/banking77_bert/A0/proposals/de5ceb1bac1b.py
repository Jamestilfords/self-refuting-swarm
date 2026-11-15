import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual norm measures encoder components unsupported by the principal in-scope feature subspace, while energy captures classifier uncertainty. A train-fitted robust scale makes the two signals comparable without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)

    mean = np.mean(x_train, axis=0)
    centered_train = x_train - mean
    centered_eval = x_eval - mean

    _, singular_values, vt = np.linalg.svd(centered_train, full_matrices=False)
    rank = min(128, vt.shape[0])
    basis = vt[:rank]

    train_proj = centered_train @ basis.T
    eval_proj = centered_eval @ basis.T
    train_residual = np.sqrt(
        np.maximum(
            np.sum(centered_train * centered_train, axis=1)
            - np.sum(train_proj * train_proj, axis=1),
            0.0,
        )
    )
    eval_residual = np.sqrt(
        np.maximum(
            np.sum(centered_eval * centered_eval, axis=1)
            - np.sum(eval_proj * eval_proj, axis=1),
            0.0,
        )
    )

    energy_train = ctx.h.energy(tr.logits)
    energy_eval = ctx.h.energy(ev.logits)

    res_med = np.median(train_residual)
    res_scale = np.median(np.abs(train_residual - res_med)) + 1e-6
    ene_med = np.median(energy_train)
    ene_scale = np.median(np.abs(energy_train - ene_med)) + 1e-6

    residual_score = (eval_residual - res_med) / res_scale
    energy_score = (energy_eval - ene_med) / ene_scale
    return residual_score + energy_score
