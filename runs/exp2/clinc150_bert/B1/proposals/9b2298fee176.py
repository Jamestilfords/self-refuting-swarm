import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples assigned to the same predicted intent tend to have characteristic top-two logit margins, whereas ambiguous OOS queries often have unusually small margins. Predicted-class train calibration accounts for intent-specific confidence scales."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margin_and_pred(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        pred = np.argmax(logits, axis=1)
        return margin, pred

    train_margin, train_pred = margin_and_pred(tr.logits)
    eval_margin, eval_pred = margin_and_pred(ev.logits)

    global_med = np.median(train_margin)
    global_mad = np.median(np.abs(train_margin - global_med))
    global_scale = max(1.4826 * global_mad, 1e-6)

    n_classes = tr.logits.shape[1]
    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_margin[train_pred == c]
        if values.size >= 8:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            medians[c] = med
            scales[c] = max(1.4826 * mad, 0.1 * global_scale, 1e-6)

    standardized_deficit = (medians[eval_pred] - eval_margin) / scales[eval_pred]
    return np.clip(standardized_deficit, -50.0, 50.0)
