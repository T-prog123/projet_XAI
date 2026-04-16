import os
import traceback

import torch
from torch import nn

from sparsify import Sae

import sae_bench.custom_saes.custom_sae_config as custom_sae_config
import sae_bench.evals.absorption.main as absorption
from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
import sae_bench.sae_bench_utils.general_utils as general_utils


repo_id = "EleutherAI/Pythia-160m-SST-k64-32k"
model_name = "pythia-160m-deduped"

hf_hookpoint = "layers.6.mlp"
tl_hook_name = "blocks.6.hook_mlp_out"
hook_layer = 6

torch_dtype = torch.float32

output_folder = "eval_results/sst"
force_rerun = True


class SparsifySSTWrapper(nn.Module):
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
            model_name=model_name, d_in=d_in, d_sae=d_sae, hook_name=tl_hook_name, hook_layer=hook_layer
        )
        self.cfg.dtype = str(torch_dtype).split(".")[-1]
        self.cfg.architecture = "sparsify_sst"
        self.cfg.training_tokens = 0
        self.cfg.device = str(device)

        self.W_enc = nn.Parameter(self.sae.encoder.weight.detach().clone().T, requires_grad=False)
        self.b_enc = nn.Parameter(self.sae.encoder.bias.detach().clone(), requires_grad=False)
        self.W_dec = nn.Parameter(self.sae.W_dec.detach().clone(), requires_grad=False)
        self.b_dec = nn.Parameter(self.sae.b_dec.detach().clone(), requires_grad=False)
        self.W_skip = nn.Parameter(self.sae.W_skip.detach().clone(), requires_grad=False) if hasattr(self.sae, "W_skip") else None

    def encode(self, x):
        top_acts, top_indices, _ = self.sae.encode(x)

        if x.dim() == 2:
            z = torch.zeros(x.shape[0], self.cfg.d_sae, device=x.device, dtype=x.dtype)
            z.scatter_(1, top_indices, top_acts)
            return z

        if x.dim() == 3:
            z = torch.zeros(x.shape[0], x.shape[1], self.cfg.d_sae, device=x.device, dtype=x.dtype)
            z.scatter_(2, top_indices, top_acts)
            return z

        raise ValueError(f"Unsupported input rank for encode: {x.dim()}")

    def decode(self, z):
        if z.dim() in (2, 3):
            return z @ self.W_dec + self.b_dec
        raise ValueError(f"Unsupported latent rank for decode: {z.dim()}")

    def forward(self, x):
        out = self.decode(self.encode(x))
        if self.W_skip is not None:
            out = out + (x @ self.W_skip)
        return out


def main():
    print("[main] starting SST absorption")
    device = general_utils.setup_environment()
    os.makedirs(output_folder, exist_ok=True)

    try:
        sae = SparsifySSTWrapper(repo_id, hf_hookpoint, tl_hook_name, model_name, hook_layer, device, torch_dtype)
    except Exception:
        print("[main] failed while building wrapper")
        traceback.print_exc()
        raise

    config = AbsorptionEvalConfig(model_name=model_name)
    config.llm_batch_size = 2
    config.llm_dtype = "float32"
    config.k_sparse_probe_batch_size = 512
    config.eval_k_sparse_probe_batch_size = 8
    config.k_sparse_probe_num_epochs = 10
    config.max_k_value = 3
    config.min_feats_for_eval = 5
    config.precalc_k_sparse_probe_sae_acts = False

    selected_saes = [("eleuther_pythia160m_sst_k64_32k_layers6mlp_absorption", sae)]

    try:
        absorption.run_eval(
            config=config,
            selected_saes=selected_saes,
            device=device,
            output_path=output_folder,
            force_rerun=force_rerun,
        )
    except Exception:
        print("[main] absorption.run_eval raised an exception")
        traceback.print_exc()
        raise

    print("[main] done")
    print(os.listdir(output_folder))


if __name__ == "__main__":
    main()