import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "Known intents tend to produce logits concentrated along one centered class direction, "
    "whereas OOS inputs often have a more diffuse centered logit shape even after removing global logit shifts."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def concentration(logits):
        z = np.asarray(logits, dtype=float)
        z = z - np.mean(z, axis=1, keepdims=True)
        norm = np.sqrt(np.sum(z * z, axis=1))
        peak = np.max(z, axis=1)
        return peak / np.maximum(norm, 1e-12)

    train_c = concentration(tr.logits)
    eval_c = concentration(ev.logits)

    center = np.median(train_c)
    scale = np.median(np.abs(train_c - center))
    scale = max(float(scale), 1e-6)

    return (center - eval_c) / scale
