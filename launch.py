# launch.py - one-cell launcher for BarameterLM.
# Colab usage (one cell):   %run launch.py
# Asks for: datasets, minutes, Hub repo, and whether/how to grow the model between datasets.
import os, sys, subprocess, random, argparse

import_msg = "Upload barameter.py to this Colab (folder icon), or push it to your GitHub repo, then run again."


def sh(cmd):
    subprocess.run(cmd, shell=True, check=False)


def ask(prompt, default=None, cast=str):
    raw = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except Exception:
        print(f"  could not read '{raw}', using {default}")
        return default


def yes(prompt, default="n"):
    return (ask(prompt + " (y/n)", default) or default).lower().startswith("y")


sh("pip -q install transformers tokenizers datasets safetensors huggingface_hub")
if not os.path.exists("barameter.py"):
    sh("wget -q https://raw.githubusercontent.com/iaBRU-Web/winter-architecture/main/barameter.py")
if not os.path.exists("barameter.py"):
    raise SystemExit(import_msg)
sys.path.insert(0, os.getcwd())
import torch
import barameter as B

dev = B.pick_device()
random.seed(0)
torch.manual_seed(0)
print(f"device: {dev}")

# ---------------- questions ----------------
raw_ds = ask("Datasets to train on, comma separated (Hugging Face names)", "yahma/alpaca-cleaned,databricks/databricks-dolly-15k")
names = [d.strip() for d in raw_ds.split(",") if d.strip()]
minutes = ask("Minutes of training per dataset", 10.0, float)
streaming = yes("Stream datasets instead of downloading them (use y for big ones)", "y")
max_examples = ask("Max examples per dataset (0 = all)", 0, int)
seq_len = ask("Sequence length", 512, int)
batch = ask("Batch size", 8, int)
base_repo = ask("Hugging Face repo to save to (blank = do not upload)", "brucoder/BarameterLM") or ""
grow_every = ask("Grow the model after every N datasets (0 = never grow)", 0, int)
add_layers = widen = 0
if grow_every:
    add_layers = ask("Layers to add per growth", 4, int)
    widen = 1 if yes("Also widen the model each growth (d_model +320)", "n") else 0

# ---------------- hub check ----------------
repo = base_repo
if repo:
    try:
        api = B.hf_api()
        me = api.whoami()
        orgs = [o.get("name") for o in me.get("orgs", [])]
        ns = repo.split("/")[0]
        print(f"logged in as: {me['name']}")
        if ns != me["name"] and ns not in orgs:
            print(f"WARNING: you are '{me['name']}' but the repo namespace is '{ns}'. Uploads would fail with 403.")
            if not yes("Continue without uploading to the Hub", "y"):
                raise SystemExit("Log in as the right account (huggingface_hub.login()) and run again.")
            repo = ""
    except SystemExit:
        raise
    except Exception as e:
        print(f"Hub login failed ({e}); continuing without upload.")
        repo = ""

# ---------------- load datasets ----------------
loaded = []
for n in names:
    try:
        mk, held = B.load_rows(n, "train", streaming, max_examples, 500)
        loaded.append((n, mk, held))
        print(f"loaded {n} ({len(held)} held-out rows)")
    except Exception as e:
        print(f"SKIPPED {n}: {e}")
if not loaded:
    raise SystemExit("No dataset could be loaded.")

# ---------------- tokenizer + model ----------------
sample = []
for n, mk, _ in loaded:
    c = 0
    for r in mk(0):
        sample.append(r)
        c += 1
        if c >= 5000:
            break
tok = B.Tok.train(sample, 16384)
model = B.BarameterForCausalLM(B.BarameterConfig(vocab_size=tok.vocab_size, max_seq_len=seq_len))
print(f"start model: {model.num_params() / 1e6:.1f}M parameters")


def grow(model, batch):
    model.eval()
    ids = torch.randint(4, model.config.vocab_size, (2, 128)).to(dev)
    with torch.no_grad():
        before = model(ids).logits
    new_d = model.config.d_model + 1 if widen else None
    grown = B.grow_model(model, add_layers=add_layers, new_d_model=new_d).to(dev).eval()
    with torch.no_grad():
        diff = (grown(ids).logits - before).abs().max().item()
    print(f"grown: {model.num_params() / 1e6:.1f}M -> {grown.num_params() / 1e6:.1f}M params, max logit diff {diff:.2e}")
    if diff > 1e-2:
        print("WARNING: growth changed the outputs more than expected.")
        if not yes("Keep the grown model anyway", "n"):
            return model, batch
    del model
    torch.cuda.empty_cache()
    if grown.num_params() > 150e6:
        grown.config.grad_checkpoint = True
        batch = max(1, batch // 2)
        print(f"large model: gradient checkpointing on, batch {batch}")
    return grown, batch


# ---------------- training loop ----------------
done = 0
for i, (name, mk, held) in enumerate(loaded):
    stage_repo = repo
    if repo and model.config.growth_history:
        stage_repo = f"{repo}-{round(model.num_params() / 1e6)}M"
    print(f"\n=== dataset {i + 1}/{len(loaded)}: {name} | {model.num_params() / 1e6:.1f}M params | repo {stage_repo or '-'} ===")
    while True:
        a = argparse.Namespace(out="barameter_run", hub_repo=stage_repo, minutes=minutes, max_steps=0, batch=batch,
                               seq_len=seq_len, lr_muon=0.02, lr_adam=1e-3, ckpt_minutes=5, tag=name.split("/")[-1],
                               log_every=20, dead_every=200, resume=False)
        try:
            B.train(model, tok, mk, held, a, dev)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch //= 2
            if batch < 1:
                raise
            print(f"out of GPU memory: retrying this dataset with batch {batch}")
    done += 1
    if grow_every and done % grow_every == 0 and i < len(loaded) - 1:
        model, batch = grow(model, batch)

print(f"\nfinished: {model.num_params() / 1e6:.1f}M parameters, store {model.mem.store.n} slots, index {model.mem.index.n} chunks")
print("datasets in memory:", model.mem.counts())

# ---------------- chat ----------------
if yes("Chat with it now", "y"):
    print("empty line to stop")
    while True:
        q = input("you> ").strip()
        if not q:
            break
        ans, info = B.answer(model, tok, q, dev)
        print(f"[score {info['best_score']} | entropy {info['entropy']:.2f} | deep {'yes' if info['deep'] else 'no'}]")
        print(ans)
