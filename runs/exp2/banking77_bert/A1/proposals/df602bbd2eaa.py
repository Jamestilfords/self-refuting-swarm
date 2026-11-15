import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual norm measures representation energy outside the dominant in-scope encoder subspace, where unfamiliar queries should retain more variation. A train-calibrated energy term preserves classifier uncertainty when residual novelty is weak."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    mean = np.mean(tr.pooled, axis=0, keepdims=True)
    x_train = tr.pooled - mean
    x_eval = ev.pooled - mean

    pca = ctx.h.pca(x_train, dim=256)
    z_train = pca.transform(x_train)
    z_eval = pca.transform(x_eval)

    train_total = np.sum(x_train * x_train, axis=1)
    eval_total = np.sum(x_eval * x_eval, axis=1)
    train_proj = np.sum(z_train * z_train, axis=1)
    eval_proj = np.sum(z_eval * z_eval, axis=1)

    train_residual = np.sqrt(np.maximum(train_total - train_proj, 0.0))
    eval_residual = np.sqrt(np.maximum(eval_total - eval_proj, 0.0))

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    residual_scale = np.std(train_residual) + 1e-8
    energy_scale = np.std(train_energy) + 1e-8
    weight = energy_scale / residual_scale

    scores = eval_energy + weight * eval_residual
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
