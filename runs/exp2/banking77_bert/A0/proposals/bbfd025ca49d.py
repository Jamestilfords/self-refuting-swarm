import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Tempered posterior entropy measures how broadly classifier evidence is distributed across intents after setting the temperature from train-time logit gaps. OOS queries should retain more competing intent mass and therefore have higher entropy than sharply concentrated in-scope predictions."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0033"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)

    train_top2 = np.partition(train_logits, -2, axis=1)[:, -2:]
    train_gap = train_top2[:, 1] - train_top2[:, 0]
    temperature = np.median(train_gap)
    temperature = max(float(temperature), 1e-3)

    z = eval_logits / temperature
    z = z - np.max(z, axis=1, keepdims=True)
    z = np.clip(z, -80.0, 0.0)
    p = np.exp(z)
    p /= np.sum(p, axis=1, keepdims=True)

    entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)

    weights = np.linspace(-1.0, 1.0, eval_logits.shape[1])
    tie_break = np.tanh((eval_logits @ weights) / (temperature * eval_logits.shape[1]))
    return entropy + 1e-9 * tie_break
