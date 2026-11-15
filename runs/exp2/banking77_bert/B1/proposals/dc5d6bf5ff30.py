import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Known intents tend to induce class-dependent logit magnitudes, whereas OOS queries often have unusually weak total classifier activation even when their predicted class is confident relative to the alternatives. Calibrating logit norm within each predicted intent avoids confusing naturally lower-scale classes with OOS inputs."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    tr_norm = np.sqrt(np.sum(tr.logits * tr.logits, axis=1))
    ev_norm = np.sqrt(np.sum(ev.logits * ev.logits, axis=1))

    global_center = float(np.median(tr_norm))
    global_scale = float(np.median(np.abs(tr_norm - global_center)))
    global_scale = max(global_scale, 1e-6)

    centers = np.full(k, global_center, dtype=float)
    scales = np.full(k, global_scale, dtype=float)

    for cls in range(k):
        values = tr_norm[tr.labels == cls]
        if values.size:
            center = float(np.median(values))
            spread = float(np.median(np.abs(values - center)))
            centers[cls] = center
            scales[cls] = max(spread, global_scale * 0.05, 1e-6)

    predicted = np.argmax(ev.logits, axis=1)
    return (centers[predicted] - ev_norm) / scales[predicted]
