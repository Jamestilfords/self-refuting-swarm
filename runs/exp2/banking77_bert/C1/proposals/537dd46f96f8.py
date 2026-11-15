import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples tend to have a clear separation between their two most competitive logits, while OOS queries often produce a smaller top-two margin. Scaling the margin by logit dispersion reduces sensitivity to global confidence shifts."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)

    train_sorted = np.sort(train_logits, axis=1)
    margins = train_sorted[:, -1] - train_sorted[:, -2]
    scale = np.median(np.abs(margins - np.median(margins))) * 1.4826
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = np.std(margins)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0

    eval_sorted = np.sort(eval_logits, axis=1)
    eval_margin = eval_sorted[:, -1] - eval_sorted[:, -2]
    centered = eval_logits - np.mean(eval_logits, axis=1, keepdims=True)
    dispersion = np.sqrt(np.mean(centered * centered, axis=1))
    train_centered = train_logits - np.mean(train_logits, axis=1, keepdims=True)
    train_dispersion = np.sqrt(np.mean(train_centered * train_centered, axis=1))
    disp_scale = np.median(train_dispersion)
    if not np.isfinite(disp_scale) or disp_scale <= 1e-12:
        disp_scale = 1.0

    normalized_margin = eval_margin / (dispersion + 0.1 * disp_scale)
    return -normalized_margin / scale
