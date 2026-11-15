import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "After removing per-item logit offset and scale, in-scope predictions should retain "
    "a characteristic robustness to fixed small perturbations of the full rival-logit pattern. "
    "OOS queries can have unusually fragile winner support across many competing intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    rng = np.random.default_rng(271828)

    k = tr.logits.shape[1]
    n_dirs = 16
    directions = rng.normal(size=(n_dirs, k))
    directions -= directions.mean(axis=1, keepdims=True)
    directions /= np.sqrt(np.sum(directions * directions, axis=1, keepdims=True)) + 1e-12
    step = 0.65

    def normalized_logits(logits):
        centered = logits - logits.mean(axis=1, keepdims=True)
        scale = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
        return centered / np.maximum(scale, 1e-6)

    def perturbation_loss(logits):
        z = normalized_logits(logits)
        winner = np.argmax(logits, axis=1)
        rows = np.arange(logits.shape[0])
        total = np.zeros(logits.shape[0], dtype=float)

        for direction in directions:
            q = z + step * direction[None, :]
            q -= np.max(q, axis=1, keepdims=True)
            log_denom = np.log(np.sum(np.exp(q), axis=1))
            total += log_denom - q[rows, winner]

        return total / float(n_dirs)

    train_loss = perturbation_loss(tr.logits)
    eval_loss = perturbation_loss(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    counts = np.bincount(train_pred, minlength=k).astype(float)
    sums = np.bincount(train_pred, weights=train_loss, minlength=k)
    global_mean = float(np.mean(train_loss))
    class_mean = np.where(counts > 0, sums / np.maximum(counts, 1.0), global_mean)

    residual = train_loss - class_mean[train_pred]
    sq_sums = np.bincount(train_pred, weights=residual * residual, minlength=k)
    global_std = float(np.std(train_loss))
    class_std = np.sqrt(sq_sums / np.maximum(counts, 1.0))
    class_std = np.where(
        (counts >= 5) & (class_std > 1e-6),
        class_std,
        max(global_std, 1e-6),
    )

    return (eval_loss - class_mean[eval_pred]) / class_std[eval_pred]
