import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "OOS inputs can be assigned a confident class while producing a centered logit profile unlike that class's in-scope logit prototypes. Cosine discrepancy to the predicted-class prototype measures this shape mismatch independently of additive logit shifts."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    train_centered = train_logits - train_logits.mean(axis=1, keepdims=True)
    eval_centered = eval_logits - eval_logits.mean(axis=1, keepdims=True)

    k = train_logits.shape[1]
    global_proto = train_centered.mean(axis=0)
    prototypes = np.empty((k, train_logits.shape[1]), dtype=float)

    for c in range(k):
        members = train_centered[labels == c]
        prototypes[c] = members.mean(axis=0) if len(members) else global_proto

    proto_norm = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes = prototypes / np.maximum(proto_norm, 1e-12)

    predicted = np.argmax(eval_logits, axis=1)
    eval_norm = np.linalg.norm(eval_centered, axis=1)
    eval_unit = eval_centered / np.maximum(eval_norm[:, None], 1e-12)

    similarity = np.sum(eval_unit * prototypes[predicted], axis=1)
    return np.nan_to_num(1.0 - similarity, nan=1.0, posinf=2.0, neginf=0.0)
