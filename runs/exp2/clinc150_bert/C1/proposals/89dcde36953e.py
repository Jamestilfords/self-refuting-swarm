import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Centered logit norm measures the concentration of evidence across intents while removing the arbitrary common logit offset. Class-conditional deviations from the training distribution identify unusually flat or unusually sharp logit shapes that can indicate OOS inputs."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    train_centered = train_logits - train_logits.mean(axis=1, keepdims=True)
    eval_centered = eval_logits - eval_logits.mean(axis=1, keepdims=True)
    train_norm = np.sqrt(np.sum(train_centered * train_centered, axis=1))
    eval_norm = np.sqrt(np.sum(eval_centered * eval_centered, axis=1))

    n_classes = train_logits.shape[1]
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    sums = np.bincount(labels, weights=train_norm, minlength=n_classes)
    global_mean = float(train_norm.mean())
    means = np.divide(
        sums, counts, out=np.full(n_classes, global_mean), where=counts > 0
    )

    residual = train_norm - means[labels]
    sq_sums = np.bincount(
        labels, weights=residual * residual, minlength=n_classes
    )
    global_scale = float(np.sqrt(np.mean((train_norm - global_mean) ** 2)))
    global_scale = max(global_scale, 1e-6)
    scales = np.sqrt(
        np.divide(
            sq_sums,
            counts,
            out=np.full(n_classes, global_scale * global_scale),
            where=counts > 1,
        )
    )
    scales = np.maximum(scales, 1e-6)

    predicted = np.argmax(eval_logits, axis=1)
    z = (eval_norm - means[predicted]) / scales[predicted]
    return z * z
