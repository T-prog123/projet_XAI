# train_sae_and_transcoder.py

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize


model_name = "HuggingFaceTB/SmolLM2-135M"
dataset_name = "wikitext"
dataset_config = "wikitext-2-raw-v1"

# load a small dataset split
dataset = load_dataset(dataset_name, dataset_config, split="train")

# keep it small so it runs quickly
dataset = dataset.select(range(2000))

tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenize into fixed-length chunks
tokenized = chunk_and_tokenize(
    dataset,
    tokenizer,
    max_seq_len=128,
    text_key="text",
    num_proc=1,
)

# GPU model load
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

gpt = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cuda"} if torch.cuda.is_available() else {"": "cpu"},
    torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
)

# -----------------------------
# 1) Train a small SAE
# -----------------------------
sae_cfg = SaeConfig(
    expansion_factor=8,   # smaller than default 32 for a cheap demo
    k=16,
    transcode=False,
)

train_cfg_sae = TrainConfig(
    sae=sae_cfg,
    batch_size=8,
    grad_acc_steps=1,
    micro_acc_steps=1,
    layers=[0],                 # only first transformer block
    save_every=50,
    save_best=False,
    log_to_wandb=False,         # disable wandb
    run_name="sae_small_demo",
    save_dir="outputs",
)

trainer_sae = Trainer(train_cfg_sae, tokenized, gpt)
trainer_sae.fit()

# -----------------------------
# 2) Train a small transcoder
# -----------------------------
transcoder_cfg = SaeConfig(
    expansion_factor=8,
    k=16,
    transcode=True,
)

train_cfg_transcoder = TrainConfig(
    sae=transcoder_cfg,
    batch_size=8,
    grad_acc_steps=1,
    micro_acc_steps=1,
    layers=[0],                 # same layer for demo
    save_every=50,
    save_best=False,
    log_to_wandb=False,
    run_name="transcoder_small_demo",
    save_dir="outputs",
)

trainer_transcoder = Trainer(train_cfg_transcoder, tokenized, gpt)
trainer_transcoder.fit()