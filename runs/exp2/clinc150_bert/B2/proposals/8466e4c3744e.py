import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "A dominant-intent concentration score compares the largest centered logit with the overall centered-logit dispersion, detecting queries whose activation is spread across many intents even when their raw logit scale is high. Predicted-class empirical calibration learns only in-scope shape variability for each assigned intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    tr_logits = np.asarray(tr.logits, dtype=np.float64)
    ev_logits = np.asarray(ev.logits, dtype=np.float64)

    def concentration(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        scale = np.sqrt(np.mean(centered * centered, axis=1))
        peak = np.max(centered, axis=1)
        return peak / np.maximum(scale, 1e-12)

    tr_conc = concentration(tr_logits)
    ev_conc = concentration(ev_logits)
    tr_pred = np.argmax(tr_logits, axis=1)
    ev_pred = np.argmax(ev_logits, axis=1)

    n_classes = tr_logits.shape[1]
    global_reference = np.sort(tr_conc)
    scores = np.empty(ev_conc.shape[0], dtype=np.float64)

    for c in range(n_classes):
        eval_mask = ev_pred == c
        if not np.any(eval_mask):
            continue

        class_values = tr_conc[tr_pred == c]
        reference = np.sort(class_values) if class_values.size >= 12 else global_reference
        rank = np.searchsorted(reference, ev_conc[eval_mask], side="right")
        tail = 1.0 - rank / (reference.size + 1.0)

        scores[eval_mask] = tail + 1e-7 * (1.0 / np.maximum(ev_conc[eval_mask], 1e-12))

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
