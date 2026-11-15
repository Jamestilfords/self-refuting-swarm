import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "Known intents produce characteristic directions in centered logit space, reflecting their full "
    "winner-versus-rival competition pattern. Angular departure from the training logit-shape prototype "
    "for the classifier-selected intent flags atypical OOS competition independently of logit scale."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0164"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def centered_unit(z):
        z = np.asarray(z, dtype=float)
        z = z - np.mean(z, axis=1, keepdims=True)
        norms = np.sqrt(np.sum(z * z, axis=1, keepdims=True))
        return z / np.maximum(norms, 1e-12)

    train_u = centered_unit(tr.logits)
    eval_u = centered_unit(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_proto = np.mean(train_u, axis=0)
    global_proto /= max(np.linalg.norm(global_proto), 1e-12)

    prototypes = np.empty((k, train_u.shape[1]), dtype=float)
    thresholds = np.empty(k, dtype=float)

    for c in range(k):
        mask = train_pred == c
        if np.any(mask):
            proto = np.mean(train_u[mask], axis=0)
            proto /= max(np.linalg.norm(proto), 1e-12)
            distances = 1.0 - np.clip(train_u[mask] @ proto, -1.0, 1.0)
            thresholds[c] = np.quantile(distances, 0.90)
        else:
            proto = global_proto
            distances = 1.0 - np.clip(train_u @ proto, -1.0, 1.0)
            thresholds[c] = np.quantile(distances, 0.90)
        prototypes[c] = proto

    angular_distance = 1.0 - np.clip(
        np.sum(eval_u * prototypes[eval_pred], axis=1), -1.0, 1.0
    )
    scale = np.maximum(thresholds[eval_pred], 1e-8)
    return angular_distance / scale
