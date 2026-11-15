import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The increase in predictive entropy when logits are softened from a sharp to a warm "
    "temperature measures spectral temperature-susceptibility beyond MSP or energy. "
    "Diffuse OOS logit spectra tend to have a smaller entropy change than supported in-scope predictions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def entropy_at_temperature(logits, temperature):
        z = logits / temperature
        z = z - np.max(z, axis=1, keepdims=True)
        p = np.exp(z)
        p /= np.sum(p, axis=1, keepdims=True)
        return -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)

    def susceptibility(logits):
        sharp = entropy_at_temperature(logits, 0.5)
        warm = entropy_at_temperature(logits, 2.0)
        return warm - sharp

    tr_gap = susceptibility(tr.logits)
    ev_gap = susceptibility(ev.logits)
    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    out = np.empty(ev_gap.shape[0], dtype=float)
    global_sorted = np.sort(tr_gap)

    for c in range(tr.logits.shape[1]):
        mask = tr_pred == c
        test_mask = ev_pred == c
        if not np.any(test_mask):
            continue
        ref = np.sort(tr_gap[mask]) if np.sum(mask) >= 8 else global_sorted
        ranks = np.searchsorted(ref, ev_gap[test_mask], side="right")
        out[test_mask] = 1.0 - (ranks + 0.5) / (ref.size + 1.0)

    return np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
