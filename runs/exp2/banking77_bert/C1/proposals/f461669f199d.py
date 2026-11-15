import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "A standardized logit concentration score measures whether one class direction dominates after accounting for each class's typical training-logit scale. OOS inputs should have less concentrated standardized evidence, while a small norm term distinguishes diffuse low-evidence predictions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)

    mean = np.mean(train_logits, axis=0)
    scale = np.std(train_logits, axis=0)
    scale = np.maximum(scale, 1e-6)

    train_z = (train_logits - mean) / scale
    train_z = train_z - np.mean(train_z, axis=1, keepdims=True)
    train_norm = np.linalg.norm(train_z, axis=1)

    norm_med = np.median(train_norm)
    norm_scale = np.quantile(train_norm, 0.9) - np.quantile(train_norm, 0.1)
    norm_scale = max(float(norm_scale), 1e-6)

    z = (eval_logits - mean) / scale
    z = z - np.mean(z, axis=1, keepdims=True)
    norm = np.linalg.norm(z, axis=1)
    top = np.max(z, axis=1)

    concentration = top / np.maximum(norm, 1e-6)
    low_evidence = (norm_med - norm) / norm_scale

    scores = -concentration + 0.03 * low_evidence
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
