import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "After removing the logit offset and scale, in-scope queries assigned to an intent should retain "
    "a class-typical relative logit shape. OOS queries can have plausible confidence yet an unusual "
    "pattern of competing intent logits, producing larger angular deviation from the predicted-class prototype."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        return centered / np.maximum(norms, 1e-12)

    train_shape = shape(tr.logits)
    eval_shape = shape(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_proto = np.mean(train_shape, axis=0)
    global_proto /= max(np.linalg.norm(global_proto), 1e-12)

    prototypes = np.empty((n_classes, tr.logits.shape[1]), dtype=float)
    sorted_distances = []

    global_dist = 1.0 - np.clip(train_shape @ global_proto, -1.0, 1.0)
    for c in range(n_classes):
        mask = train_pred == c
        if np.any(mask):
            proto = np.mean(train_shape[mask], axis=0)
            proto /= max(np.linalg.norm(proto), 1e-12)
            distances = 1.0 - np.clip(train_shape[mask] @ proto, -1.0, 1.0)
        else:
            proto = global_proto
            distances = global_dist
        prototypes[c] = proto
        sorted_distances.append(np.sort(distances))

    raw = 1.0 - np.sum(eval_shape * prototypes[eval_pred], axis=1)
    scores = np.empty(raw.shape[0], dtype=float)

    for c in range(n_classes):
        mask = eval_pred == c
        if np.any(mask):
            reference = sorted_distances[c]
            ranks = np.searchsorted(reference, raw[mask], side="right")
            scores[mask] = ranks / (reference.size + 1.0)

    return np.clip(scores, 0.0, 1.0)
