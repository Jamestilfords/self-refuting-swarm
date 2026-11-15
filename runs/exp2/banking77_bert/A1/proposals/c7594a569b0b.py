import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The centered logit-vector norm measures overall class-separation strength while being invariant to additive logit shifts. Class-conditional robust calibration flags queries whose logit concentration is atypically weak or unusually extreme for their predicted intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0011"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def concentration(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        return np.sqrt(np.sum(centered * centered, axis=1))

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_value = concentration(tr.logits)
    eval_value = concentration(ev.logits)

    global_med = np.median(train_value)
    global_mad = np.median(np.abs(train_value - global_med))
    global_scale = max(1.4826 * global_mad, 1e-6)

    n_classes = tr.logits.shape[1]
    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_value[train_pred == c]
        if values.size >= 4:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            scale = max(1.4826 * mad, 0.1 * global_scale, 1e-6)
            weight = values.size / (values.size + 12.0)
            medians[c] = weight * med + (1.0 - weight) * global_med
            scales[c] = weight * scale + (1.0 - weight) * global_scale

    z = np.abs(eval_value - medians[eval_pred]) / scales[eval_pred]
    return np.log1p(z)
