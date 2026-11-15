import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "After removing the logit offset and scale, each predicted intent has a characteristic "
    "angular pattern of support across its competing rivals. A class-conditioned surprise score "
    "on a fixed random sketch of that normalized competition pattern detects unfamiliar logit shapes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0190", "m_0164"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)
    n_classes = train_logits.shape[1]

    def angular_logits(x):
        centered = x - np.mean(x, axis=1, keepdims=True)
        norms = np.sqrt(np.sum(centered * centered, axis=1, keepdims=True))
        return centered / np.maximum(norms, 1e-8)

    train_u = angular_logits(train_logits)
    eval_u = angular_logits(eval_logits)

    sketch_dim = min(24, n_classes)
    rng = np.random.default_rng(271828)
    raw = rng.normal(size=(n_classes, sketch_dim))
    q, _ = np.linalg.qr(raw)
    q = q[:, :sketch_dim]

    train_z = train_u @ q
    eval_z = eval_u @ q

    train_pred = np.argmax(train_logits, axis=1)
    eval_pred = np.argmax(eval_logits, axis=1)

    global_mean = np.mean(train_z, axis=0)
    global_var = np.var(train_z, axis=0) + 1e-5

    means = np.empty((n_classes, sketch_dim), dtype=float)
    variances = np.empty((n_classes, sketch_dim), dtype=float)

    for c in range(n_classes):
        members = train_z[train_pred == c]
        count = members.shape[0]
        if count == 0:
            means[c] = global_mean
            variances[c] = global_var
            continue

        local_mean = np.mean(members, axis=0)
        strength = count / (count + 12.0)
        means[c] = strength * local_mean + (1.0 - strength) * global_mean

        local_var = np.mean((members - local_mean) ** 2, axis=0)
        variances[c] = strength * local_var + (1.0 - strength) * global_var + 1e-5

    diff = eval_z - means[eval_pred]
    scores = np.sum((diff * diff) / variances[eval_pred], axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
