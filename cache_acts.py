"""Cache Llama 3.1 8B Instruct last-prompt-token activations on Modal.

  modal run cache_acts.py::orchestrate --limit 200      # thin slice first (~2 min)
  modal run --detach cache_acts.py::orchestrate         # full WildJailbreak eval set (2,210 rows)
  modal volume ls hack-cache /acts/wjb_eval

Output on the volume, rows in the same order in every file (sorted by id):
  /acts/<run>/L{layer}.npy   float16 [N, 4096]
  /acts/<run>/meta.parquet   id, source, label (1 = harmful), data_type, prompt, n_tokens, act_norm_L{layer}
"""
import modal

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LAYERS = [8, 12, 16, 20, 24, 28]      # block indices; hidden_states[l + 1] is block l's output
BATCH, SHARD = 16, 256

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers>=5.5", "accelerate", "huggingface_hub", "numpy",
                 "pandas", "pyarrow", "git+https://github.com/anthropics/jacobian-lens")
    .env({"HF_HOME": "/cache/hf"})
)
vol = modal.Volume.from_name("hack-cache", create_if_missing=True)
app = modal.App("cache-acts", image=image)
common = dict(volumes={"/cache": vol}, secrets=[modal.Secret.from_dotenv()])


@app.cls(gpu="H100", timeout=3600, max_containers=4, **common)
class Cacher:
    @modal.enter()
    def load(self):
        import torch, transformers
        self.tok = transformers.AutoTokenizer.from_pretrained(MODEL)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.tok.padding_side = "left"            # position -1 is the real last token in every row
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    @modal.method()
    def run_shard(self, shard: dict) -> str:
        import os, numpy as np, torch
        out = f"/cache/acts/{shard['run']}/shards/{shard['name']}.npz"
        if os.path.exists(out):
            return f"skip {shard['name']}"
        acts = {l: [] for l in LAYERS}
        n_tokens = []
        for i in range(0, len(shard["prompts"]), BATCH):
            texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                  add_generation_prompt=True)
                     for p in shard["prompts"][i:i + BATCH]]
            enc = self.tok(texts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to("cuda")   # template already adds BOS
            with torch.no_grad():
                hs = self.model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            for l in LAYERS:
                acts[l].append(hs[l + 1][:, -1, :].float().cpu().numpy())
            n_tokens += enc.attention_mask.sum(1).tolist()
            del hs
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, ids=np.array(shard["ids"]), n_tokens=np.array(n_tokens),
                 **{f"L{l}": np.concatenate(a).astype(np.float16) for l, a in acts.items()})
        vol.commit()
        return f"ok {shard['name']} ({len(shard['ids'])} rows)"


@app.function(timeout=4 * 3600, **common)
def orchestrate(limit: int = 0, run: str = "wjb_eval"):
    import glob, os, numpy as np, pandas as pd
    from huggingface_hub import hf_hub_download

    raw = pd.read_csv(hf_hub_download("allenai/wildjailbreak", "eval/eval.tsv", repo_type="dataset"), sep="\t")
    print("columns:", list(raw.columns), "| rows:", len(raw))
    df = pd.DataFrame({"id": [f"{run}_{i:05d}" for i in range(len(raw))], "source": "wildjailbreak_eval",
                       "label": raw["label"].astype(int), "data_type": raw["data_type"],
                       "prompt": raw["adversarial"].astype(str)})
    if limit:
        df = df.sample(n=limit, random_state=0)
    df = df.sort_values("id").reset_index(drop=True)

    by_len = df.assign(n=df.prompt.str.len()).sort_values("n")     # similar lengths together: less padding
    shards = [{"run": run, "name": f"s{k:04d}", "ids": c.id.tolist(), "prompts": c.prompt.tolist()}
              for k, c in enumerate(by_len.iloc[i:i + SHARD] for i in range(0, len(by_len), SHARD))]
    for msg in Cacher().run_shard.map(shards, order_outputs=False):
        print(msg)

    vol.reload()
    parts = [np.load(p) for p in sorted(glob.glob(f"/cache/acts/{run}/shards/*.npz"))]
    ids = np.concatenate([p["ids"] for p in parts])
    order = pd.Series(np.arange(len(ids)), index=ids).loc[df.id].to_numpy()   # match by id, never by position
    assert len(order) == len(df) and (ids[order] == df.id.to_numpy()).all()
    df["n_tokens"] = np.concatenate([p["n_tokens"] for p in parts])[order]
    for l in LAYERS:
        a = np.concatenate([p[f"L{l}"] for p in parts])[order]
        assert np.isfinite(a).all(), f"non-finite values at L{l}"
        np.save(f"/cache/acts/{run}/L{l}.npy", a)
        df[f"act_norm_L{l}"] = np.linalg.norm(a.astype(np.float32), axis=1)
    df.to_parquet(f"/cache/acts/{run}/meta.parquet", index=False)
    vol.commit()
    print(df.groupby(["data_type", "label"]).size())
    print(df[["n_tokens"] + [f"act_norm_L{l}" for l in LAYERS]].describe().round(1))
    print(f"DONE: {len(df)} rows -> /acts/{run}/")
