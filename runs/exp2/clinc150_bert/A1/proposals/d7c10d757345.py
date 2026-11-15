import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The top-two margin normalized by centered logit scale measures relative winner separation rather than raw confidence. "
    "Calibrating this quantity per intent makes unusually weak competition margins receive higher OOS scores."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    def relative_margin(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        scale = np.sqrt(np.mean(centered * centered, axis=1)) + eps
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = np.max(top2, axis=1) - np.min(top2, axis=1)
        return margin / scale

    train_margin = relative_margin(tr.logits)
    global_med = np.median(train_margin)
    global_mad = np.median(np.abs(train_margin - global_med)) + eps

    k = tr.logits.shape[1]
    med = np.full(k, global_med, dtype=float)
    mad = np.full(k, global_mad, dtype=float)

    for c in range(k):
        values = train_margin[tr.labels == c]
        if values.size >= 4:
            center = np.median(values)
            spread = np.median(np.abs(values - center))
            med[c] = center
            mad[c] = max(spread, 0.05 * global_mad, eps)

    pred = np.argmax(ev.logits, axis=1)
    eval_margin = relative_margin(ev.logits)
    scores = (med[pred] - eval_margin) / mad[pred]
    return np.clip(scores, -1e6, 1e6)
