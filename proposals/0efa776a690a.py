import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The difference between energy at two temperatures is invariant to a common logit shift "
    "and measures how concentrated the winning logit is relative to its competitors. "
    "Diffuse or atypical competition patterns receive a high predicted-class-calibrated tail score."
)
SOURCE = "https://proceedings.iclr.cc/paper_files/paper/2025/file/7453c6d7e78c1cc6e431cdf6d3be52a2-Paper-Conference.pdf"
PARENTS = []
MEMORY_USED = ["m_0111"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def tempered_energy(logits, temperature):
        x = np.nan_to_num(
            np.asarray(logits, dtype=float),
            nan=0.0,
            posinf=50.0,
            neginf=-50.0,
        )
        z = x / temperature
        m = np.max(z, axis=1, keepdims=True)
        return -temperature * (
            m[:, 0] + np.log(np.sum(np.exp(z - m), axis=1))
        )

    def curvature(logits):
        # Large values indicate that the logits retain substantial multi-class
        # competition when the temperature is increased.
        return tempered_energy(logits, 1.0) - tempered_energy(logits, 2.0)

    train_curve = curvature(tr.logits)
    eval_curve = curvature(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_sorted = np.sort(train_curve)
    per_class = []
    for c in range(n_classes):
        values = train_curve[train_pred == c]
        per_class.append(np.sort(values) if values.size >= 8 else global_sorted)

    global_med = np.median(train_curve)
    global_scale = np.median(np.abs(train_curve - global_med)) * 1.4826
    global_scale = max(float(global_scale), 1e-6)

    out = np.empty(eval_curve.shape[0], dtype=float)
    for c in range(n_classes):
        mask = eval_pred == c
        if not np.any(mask):
            continue
        reference = per_class[c]
        ranks = np.searchsorted(reference, eval_curve[mask], side="right")
        tail = (ranks + 1.0) / (reference.size + 2.0)
        smooth = 1.0 / (
            1.0 + np.exp(-np.clip((eval_curve[mask] - global_med) / global_scale, -30.0, 30.0))
        )
        out[mask] = tail + 0.02 * smooth

    return np.nan_to_num(out, nan=0.0, posinf=1.02, neginf=0.0)
