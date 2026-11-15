import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Classifier-predicted class conditioning turns low-variance residual distance into a "
    "classifier-feature disagreement signal. Near-OOS queries assigned confidently to an intent "
    "but deviating along tightly constrained within-class directions receive high scores."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0010", "m_0012"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    n_classes = tr.logits.shape[1]
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    means = np.zeros((n_classes, x.shape[1]), dtype=np.float64)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(x.shape[0] - n_classes, 1)
    vals, vecs = np.linalg.eigh(cov)

    positive = vals[vals > 1e-10]
    floor = max(float(np.median(positive)) * 1e-4, 1e-8) if positive.size else 1e-8
    n_tail = min(64, x.shape[1])
    tail_vecs = vecs[:, :n_tail]
    tail_vals = np.maximum(vals[:n_tail], floor)

    pred = np.argmax(ev.logits, axis=1)
    residual = z - means[pred]
    projected = residual @ tail_vecs
    scores = np.sum((projected * projected) / tail_vals[None, :], axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
