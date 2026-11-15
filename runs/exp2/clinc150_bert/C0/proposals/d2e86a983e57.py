import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Out-of-scope inputs often retain classifier confidence while having a large encoder-feature component outside the dominant in-scope principal subspace. Combining this residual norm with energy captures both representation shift and logit uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    mean = x_train.mean(axis=0)
    centered_train = x_train - mean
    centered_eval = x_eval - mean

    cov = centered_train.T @ centered_train / max(len(centered_train) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    n_components = min(128, eigvecs.shape[1])
    principal = eigvecs[:, -n_components:]

    train_proj_sq = np.sum((centered_train @ principal) ** 2, axis=1)
    eval_proj_sq = np.sum((centered_eval @ principal) ** 2, axis=1)

    train_total_sq = np.sum(centered_train ** 2, axis=1)
    eval_total_sq = np.sum(centered_eval ** 2, axis=1)

    train_residual = np.sqrt(np.maximum(train_total_sq - train_proj_sq, 0.0))
    eval_residual = np.sqrt(np.maximum(eval_total_sq - eval_proj_sq, 0.0))

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    residual_scale = np.std(train_residual) + 1e-8
    energy_scale = np.std(train_energy) + 1e-8

    residual_z = (eval_residual - np.mean(train_residual)) / residual_scale
    energy_z = (eval_energy - np.mean(train_energy)) / energy_scale

    return np.nan_to_num(residual_z + energy_z, nan=0.0, posinf=1e6, neginf=-1e6)
