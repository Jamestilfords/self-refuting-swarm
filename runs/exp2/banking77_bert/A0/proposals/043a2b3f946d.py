import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "After removing the arbitrary common logit offset, in-scope examples tend to have "
    "class-conditional centered-logit magnitudes typical of their predicted intent. "
    "A train-calibrated lower-tail score flags flattened or unusually weak logit shapes beyond MSP and energy."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_logits = np.asarray(tr.logits, dtype=np.float64)
    eval_logits = np.asarray(ev.logits, dtype=np.float64)

    train_centered = train_logits - train_logits.mean(axis=1, keepdims=True)
    eval_centered = eval_logits - eval_logits.mean(axis=1, keepdims=True)

    train_norm = np.sqrt(np.sum(train_centered * train_centered, axis=1))
    eval_norm = np.sqrt(np.sum(eval_centered * eval_centered, axis=1))

    train_pred = np.argmax(train_logits, axis=1)
    eval_pred = np.argmax(eval_logits, axis=1)
    n_classes = train_logits.shape[1]

    global_med = np.median(train_norm)
    global_scale = np.median(np.abs(train_norm - global_med)) + 1e-6
    class_med = np.full(n_classes, global_med, dtype=np.float64)
    class_scale = np.full(n_classes, global_scale, dtype=np.float64)

    for c in range(n_classes):
        values = train_norm[train_pred == c]
        if values.size >= 4:
            med = np.median(values)
            scale = np.median(np.abs(values - med)) + 1e-6
            class_med[c] = med
            class_scale[c] = max(scale, 0.25 * global_scale)

    z = (class_med[eval_pred] - eval_norm) / class_scale[eval_pred]
    return np.clip(z, -50.0, 50.0)
