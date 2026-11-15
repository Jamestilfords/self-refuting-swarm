import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "For confident predictions this preserves energy ranking, but for low-MSP queries it switches to the Mahalanobis distance from the class selected by the classifier. Class-conditional distance can identify feature-inconsistent uncertain queries without globally re-ranking confident items."
SOURCE = ""
PARENTS = ["energy", "msp", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_class_d = g.class_mahalanobis_sq(tr.pooled)
    eval_class_d = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_d = train_class_d[np.arange(train_pred.shape[0]), train_pred]
    eval_d = eval_class_d[np.arange(eval_pred.shape[0]), eval_pred]

    train_e = ctx.h.energy(tr.logits)
    eval_e = ctx.h.energy(ev.logits)

    d_center = np.median(train_d)
    d_scale = np.median(np.abs(train_d - d_center)) * 1.4826 + 1e-6
    e_center = np.median(train_e)
    e_scale = np.median(np.abs(train_e - e_center)) * 1.4826 + 1e-6

    distance_score = (eval_d - d_center) / d_scale
    energy_score = (eval_e - e_center) / e_scale

    train_msp = ctx.h.msp(tr.logits)
    eval_msp = ctx.h.msp(ev.logits)
    uncertainty_cutoff = np.quantile(train_msp, 0.35)

    scores = np.where(eval_msp < uncertainty_cutoff, distance_score, energy_score)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
