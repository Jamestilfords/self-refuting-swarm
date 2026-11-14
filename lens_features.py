"""Jacobian-lens feature columns from cached activations (no prompts are re-run).

  modal run lens_features.py --run wjb_eval_full
  modal volume get hack-cache /acts/wjb_eval_full/lens.parquet out/

For each lens layer, per prompt:
  lens_refuse_L{l}   probability mass on the frozen REFUSE word list
  lens_comply_L{l}   probability mass on the frozen COMPLY word list
  lens_entropy_L{l}  entropy of the lens distribution (higher = more scattered readout)
  lens_top1_L{l}     the single most likely token (for inspection and the demo)
Output: /acts/<run>/lens.parquet, keyed by id (join to meta.parquet on id).

Credits: Llama 3.1 8B Instruct (Meta); Jacobian lens code (Anthropic, Apache-2.0);
pre-fitted lens from neuronpedia/jacobian-lens.
"""
import modal

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LENS_REPO, LENS_REV = "neuronpedia/jacobian-lens", "qwen-n1000"
LENS_FILE = "llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt"
LENS_LAYERS = [16, 20, 24, 28]

# FROZEN 2026-11-14 before looking at any results. Do not edit after a run.
REFUSE = [" sorry", "Sorry", " cannot", " unable", " refuse", " illegal",
          " harmful", " dangerous", " unethical", " inappropriate", " apologize"]
COMPLY = [" Sure", "Sure", " Here", "Here", " Certainly", "Certainly", " Step", " Absolutely", " Of"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers>=5.5", "accelerate", "huggingface_hub", "numpy",
                 "pandas", "pyarrow", "git+https://github.com/anthropics/jacobian-lens")
    .env({"HF_HOME": "/cache/hf"})
)
vol = modal.Volume.from_name("hack-cache", create_if_missing=True)
app = modal.App("lens-features", image=image)


@app.function(gpu="H100", timeout=3600, volumes={"/cache": vol}, secrets=[modal.Secret.from_dotenv()])
def lens_features(run: str = "wjb_eval_full"):
    import jlens, numpy as np, pandas as pd, torch, transformers

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    lm = jlens.from_hf(model, tok)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REV)

    first = lambda words: sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})
    rid, cid = first(REFUSE), first(COMPLY)
    print("refuse tokens:", [tok.decode([t]) for t in rid])
    print("comply tokens:", [tok.decode([t]) for t in cid])

    out = pd.read_parquet(f"/cache/acts/{run}/meta.parquet", columns=["id"])
    for l in [l for l in LENS_LAYERS if l in lens.source_layers]:
        acts = torch.from_numpy(np.load(f"/cache/acts/{run}/L{l}.npy")).float()
        assert len(acts) == len(out)
        refuse, comply, entropy, top1 = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(acts), 256):
                logp = lm.unembed(lens.transport(acts[i:i + 256].cuda(), l)).float().log_softmax(-1)
                p = logp.exp()
                refuse.append(p[:, rid].sum(-1).cpu())
                comply.append(p[:, cid].sum(-1).cpu())
                entropy.append(-(p * logp).sum(-1).cpu())
                top1.append(logp.argmax(-1).cpu())
        out[f"lens_refuse_L{l}"] = torch.cat(refuse).numpy()
        out[f"lens_comply_L{l}"] = torch.cat(comply).numpy()
        out[f"lens_entropy_L{l}"] = torch.cat(entropy).numpy()
        out[f"lens_top1_L{l}"] = [tok.decode([t]) for t in torch.cat(top1).tolist()]
        print(f"L{l} most common top-1 tokens:", out[f"lens_top1_L{l}"].value_counts().head(8).to_dict())

    assert out.drop(columns="id").select_dtypes("number").notna().all().all()
    out.to_parquet(f"/cache/acts/{run}/lens.parquet", index=False)
    vol.commit()
    print(out.select_dtypes("number").describe().round(4).T[["mean", "std", "min", "max"]])
    print(f"DONE: {len(out)} rows -> /acts/{run}/lens.parquet")


@app.local_entrypoint()
def main(run: str = "wjb_eval_full"):
    lens_features.remote(run)
