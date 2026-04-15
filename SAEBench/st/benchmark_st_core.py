import os
import traceback
import json
from pathlib import Path

import torch
from torch import nn

from sparsify import Sae

import sae_bench.custom_saes.custom_sae_config as custom_sae_config
import sae_bench.evals.core.main as core
import sae_bench.sae_bench_utils.general_utils as general_utils


# config
repo_id = "EleutherAI/Pythia-160m-ST-k64-65k"
model_name = "pythia-160m-deduped"

# HF / sparsify naming
hf_hookpoint = "layers.6.mlp"

# TransformerLens / SAEBench naming
tl_hook_name = "blocks.6.hook_mlp_out"
hook_layer = 6

dataset = "Skylion007/openwebtext"
context_size = 128
eval_batch_size_prompts = 16
n_eval_reconstruction_batches = 5
n_eval_sparsity_variance_batches = 5

torch_dtype = torch.float32
str_dtype = "float32"

output_folder = "eval_results/st"


class SparsifyWrapper(nn.Module):
    def __init__(self, repo_id, hf_hookpoint, tl_hook_name, model_name, hook_layer, device, torch_dtype):
        super().__init__()

        self.sae = Sae.load_from_hub(repo_id, hookpoint=hf_hookpoint)
        self.sae = self.sae.to(device=device, dtype=torch_dtype)
        self.sae.eval()

        self.device = device
        self.dtype = torch_dtype

        d_in = self.sae.encoder.weight.shape[1]
        d_sae = self.sae.encoder.weight.shape[0]

        self.cfg = custom_sae_config.CustomSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_name=tl_hook_name,
            hook_layer=hook_layer,
        )
        self.cfg.dtype = str(torch_dtype).split(".")[-1]
        self.cfg.architecture = "sparsify_topk"
        self.cfg.training_tokens = 0
        self.cfg.device = str(device)

        # SAEBench expects W_enc as (d_in, d_sae)
        self.W_enc = nn.Parameter(self.sae.encoder.weight.detach().clone().T, requires_grad=False)
        self.b_enc = nn.Parameter(self.sae.encoder.bias.detach().clone(), requires_grad=False)
        self.W_dec = nn.Parameter(self.sae.W_dec.detach().clone(), requires_grad=False)
        self.b_dec = nn.Parameter(self.sae.b_dec.detach().clone(), requires_grad=False)

    def encode(self, x):
        top_acts, top_indices, _ = self.sae.encode(x)

        z = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.cfg.d_sae,
            device=x.device,
            dtype=x.dtype,
        )
        z.scatter_(2, top_indices, top_acts)
        return z

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def strip_eval_details(folder):
    json_files = list(Path(folder).glob("*.json"))
    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "eval_result_details" in data:
            data["eval_result_details"] = []

        stripped_path = path.with_name(path.stem + "_compact.json")
        with open(stripped_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"[main] wrote compact file: {stripped_path.name}")


def main():
    print("[main] starting ST core eval")
    print(f"[main] cuda available={torch.cuda.is_available()}")

    device = general_utils.setup_environment()
    os.makedirs(output_folder, exist_ok=True)

    try:
        sae = SparsifyWrapper(
            repo_id=repo_id,
            hf_hookpoint=hf_hookpoint,
            tl_hook_name=tl_hook_name,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device,
            torch_dtype=torch_dtype,
        )
    except Exception:
        print("[main] failed while building wrapper")
        traceback.print_exc()
        raise

    print("[main] wrapper ready")
    print(f"[main] device={device}")
    print(f"[main] repo_id={repo_id}")
    print(f"[main] hf_hookpoint={hf_hookpoint}")
    print(f"[main] tl_hook_name={tl_hook_name}")
    print(f"[main] W_enc.shape={tuple(sae.W_enc.shape)}")
    print(f"[main] W_dec.shape={tuple(sae.W_dec.shape)}")

    selected_saes = [("eleuther_pythia160m_st_k64_65k_layers6mlp", sae)]

    try:
        core.multiple_evals(
            selected_saes=selected_saes,
            n_eval_reconstruction_batches=n_eval_reconstruction_batches,
            n_eval_sparsity_variance_batches=n_eval_sparsity_variance_batches,
            eval_batch_size_prompts=eval_batch_size_prompts,
            compute_featurewise_density_statistics=True,
            compute_featurewise_weight_based_metrics=True,
            exclude_special_tokens_from_reconstruction=True,
            dataset=dataset,
            context_size=context_size,
            output_folder=output_folder,
            verbose=True,
            dtype=str_dtype,
        )
    except Exception:
        print("[main] core.multiple_evals raised an exception")
        traceback.print_exc()
        raise

    strip_eval_details(output_folder)

    print("[main] done")
    print(f"[main] output folder contents: {os.listdir(output_folder)}")


if __name__ == "__main__":
    main()