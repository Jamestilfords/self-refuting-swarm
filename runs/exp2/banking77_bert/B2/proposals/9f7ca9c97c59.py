import numpy as np
from scipy.special import logsumexp

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Tempered energy emphasizes the overall logit-spectrum shape rather than only the largest logit, making diffuse or weakly supported intent evidence more OOS-like. Train-only standardization makes the score numerically comparable to other calibrated detectors."
SOURCE = "https://proceedings.neurips.cc/paper_files/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf"
PARENTS = []
MEMORY_USED = ["m_0024"]


def score(ctx):
    temperature = 2.0
    train_energy = -temperature * logsumexp(
        ctx.train.logits / temperature, axis=1
    )
    center = float(np.mean(train_energy))
    scale = float(np.std(train_energy))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0

    eval_energy = -temperature * logsumexp(
        ctx.eval.logits / temperature, axis=1
    )
    fallback = float(np.max(np.abs(train_energy - center)) / scale + 10.0)
    return np.nan_to_num(
        (eval_energy - center) / scale,
        nan=fallback,
        posinf=fallback,
        neginf=-fallback,
    )
