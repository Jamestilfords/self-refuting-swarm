import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Out-of-scope queries often induce lower overall logit magnitude than in-scope examples, even when their top-two margin is deceptively confident. Class-conditional calibration of logit norm accounts for differing confidence scales across intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    train_norm = np.sqrt(np.sum(train_logits * train_logits, axis=1))
    eval_norm = np.sqrt(np.sum(eval_logits * eval_logits, axis=1))

    n_classes = train_logits.shape[1]
    global_mean = float(np.mean(train_norm))
    global_scale = float(np.std(train_norm))
    global_scale = max(global_scale, 1e-6)

    class_mean = np.full(n_classes, global_mean, dtype=float)
    class_scale = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_norm[labels == c]
        if values.size >= 2:
            class_mean[c] = float(np.mean(values))
            class_scale[c] = max(float(np.std(values)), 1e-6)
        elif values.size == 1:
            class_mean[c] = float(values[0])

    predicted = np.argmax(eval_logits, axis=1)
    return (class_mean[predicted] - eval_norm) / class_scale[predicted]
