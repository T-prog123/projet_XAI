# train_sae_and_transcoder.py

import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

model_name = "HuggingFaceTB/SmolLM2-135M"
dataset_name = "wikitext"
dataset_config = "wikitext-2-raw-v1"

# --- MULTI-LAYER CONFIGURATION ---
# Let's train on an early, middle, and late layer for comparison
target_layers = [0, 4, 8] 

# load a small dataset split
dataset = load_dataset(dataset_name, dataset_config, split="train")
dataset = dataset.select(range(2000)) # keep it small so it runs quickly

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized = chunk_and_tokenize(
    dataset,
    tokenizer,
    max_seq_len=128,
    text_key="text",
    num_proc=1,
)

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

gpt = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cuda"} if torch.cuda.is_available() else {"": "cpu"},
    torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
)

# -----------------------------
# 1) Train SAE
# -----------------------------
sae_cfg = SaeConfig(expansion_factor=8, k=16, transcode=False)
train_cfg_sae = TrainConfig(
    sae=sae_cfg, batch_size=8, grad_acc_steps=1, micro_acc_steps=1,
    layers=target_layers, save_every=50, save_best=False, log_to_wandb=False,
    run_name="sae_small_demo", save_dir="outputs",
)
Trainer(train_cfg_sae, tokenized, gpt).fit()

# -----------------------------
# 2) Train Transcoder
# -----------------------------
transcoder_cfg = SaeConfig(expansion_factor=8, k=16, transcode=True)
train_cfg_transcoder = TrainConfig(
    sae=transcoder_cfg, batch_size=8, grad_acc_steps=1, micro_acc_steps=1,
    layers=target_layers, save_every=50, save_best=False, log_to_wandb=False,
    run_name="transcoder_small_demo", save_dir="outputs",
)
Trainer(train_cfg_transcoder, tokenized, gpt).fit()

# -----------------------------
# 3) Train Skip Transcoder (SST)
# -----------------------------
sst_cfg = SaeConfig(expansion_factor=8, k=16, transcode=True, skip_connection=True)
train_cfg_sst = TrainConfig(
    sae=sst_cfg, batch_size=8, grad_acc_steps=1, micro_acc_steps=1,
    layers=target_layers, save_every=50, save_best=False, log_to_wandb=False,
    run_name="sst_small_demo", save_dir="outputs",
)
trainer_sst = Trainer(train_cfg_sst, tokenized, gpt)
trainer_sst.fit()

# -----------------------------
# EXPLICIT W_SKIP EXTRACTION FOR MULTIPLE LAYERS
# -----------------------------
print("\n--- Explicitly Saving W_skip Tensors ---")
os.makedirs("outputs/sst_small_demo/w_skips", exist_ok=True)

# Depending on the library version, multi-layer SAEs might be stored in trainer.saes or a ModuleDict
sae_collection = getattr(trainer_sst, 'saes', getattr(trainer_sst, 'sae', {}))

# Create an iterable of (layer_name, sae_module)
if hasattr(sae_collection, 'items'):
    sae_items = sae_collection.items()
elif isinstance(sae_collection, list):
    sae_items = zip(target_layers, sae_collection)
else:
    # Fallback if it only trained one layer despite the list
    sae_items = [(target_layers[0], sae_collection)]

for layer_id, sae_module in sae_items:
    w_skip_tensor = None
    
    # Safely extract the weights
    if hasattr(sae_module, 'W_skip'):
        w_skip_tensor = sae_module.W_skip.detach().cpu()
    elif hasattr(sae_module, 'skip'):
        w_skip_tensor = sae_module.skip.weight.detach().cpu()

    if w_skip_tensor is not None:
        save_path = f"outputs/sst_small_demo/w_skips/W_skip_layer_{layer_id}.pt"
        torch.save(w_skip_tensor, save_path)
        print(f"Layer {layer_id} -> Saved W_skip {w_skip_tensor.shape} to {save_path}")
    else:
        print(f"Warning: Could not find skip connection weights for layer {layer_id}")