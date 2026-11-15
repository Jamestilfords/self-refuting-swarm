import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Trailing within-class covariance directions capture low-variance constraints that OOS queries may violate, while class-conditional calibration prevents naturally broad intents from dominating. A small train-normalized energy term breaks ties among similarly residual-like queries using classifier uncertainty."
SOURCE = "https://arxiv.org/html/2510.15202v2"
PARENTS = ["energy"]
MEMORY_USED = ["m_0090"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    residual = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_tail = np.asarray(residual.bottom(tr.pooled), dtype=float)
    eval_tail = np.asarray(residual.bottom(ev.pooled), dtype=float)

    labels = np.asarray(tr.labels, dtype=int)
    n_classes = int(max(labels.max() + 1, tr.logits.shape[1]))
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    sums = np.bincount(labels, weights=train_tail, minlength=n_classes)
    means = sums / np.maximum(counts, 1.0)

    centered = train_tail - means[labels]
    vars_ = np.bincount(labels, weights=centered * centered,
                        minlength=n_classes) / np.maximum(counts, 1.0)
    global_scale = np.sqrt(np.mean((train_tail - np.mean(train_tail)) ** 2))
    global_scale = max(float(global_scale), 1e-6)
    scales = np.sqrt(np.maximum(vars_, (0.15 * global_scale) ** 2))

    pred = np.argmax(ev.logits, axis=1)
    tail_z = (eval_tail - means[pred]) / scales[pred]

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    energy_center = float(np.median(train_energy))
    energy_scale = float(np.median(np.abs(train_energy - energy_center)))
    energy_scale = max(1.4826 * energy_scale, 1e-6)
    energy_z = (eval_energy - energy_center) / energy_scale

    scores = tail_z + 0.12 * energy_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
