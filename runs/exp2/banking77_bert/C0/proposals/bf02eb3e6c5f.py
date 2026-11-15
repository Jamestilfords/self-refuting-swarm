import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples assigned to a class tend to share a characteristic centered logit direction, while OOS examples can have atypical relative logit patterns even when their maximum logit is large."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    train_centered = train_logits - train_logits.mean(axis=1, keepdims=True)
    train_unit = train_centered / np.maximum(
        np.linalg.norm(train_centered, axis=1, keepdims=True), 1e-12
    )

    k = train_logits.shape[1]
    prototypes = np.zeros((k, k), dtype=float)
    global_proto = train_unit.mean(axis=0)
    global_proto /= max(np.linalg.norm(global_proto), 1e-12)

    for c in range(k):
        members = train_unit[labels == c]
        if members.shape[0] == 0:
            prototypes[c] = global_proto
        else:
            proto = members.mean(axis=0)
            prototypes[c] = proto / max(np.linalg.norm(proto), 1e-12)

    eval_centered = eval_logits - eval_logits.mean(axis=1, keepdims=True)
    eval_unit = eval_centered / np.maximum(
        np.linalg.norm(eval_centered, axis=1, keepdims=True), 1e-12
    )
    predicted = np.argmax(eval_logits, axis=1)
    similarity = np.sum(eval_unit * prototypes[predicted], axis=1)
    return 1.0 - similarity
