import os
import traceback

import torch
from torch import nn

from sparsify import Sae

import sae_bench.custom_saes.custom_sae_config as custom_sae_config
import sae_bench.evals.sparse_probing.main as sparse_probing
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
import sae_bench.sae_bench_utils.activation_collection as activation_collection
import sae_bench.sae_bench_utils.general_utils as general_utils


# config
repo_id = "EleutherAI/Pythia-160m-ST-k64-65k"
model_name = "pythia-160m-deduped"

hf_hookpoint = "layers.6.mlp"
tl_hook_name = "blocks.6.hook_mlp_out"
hook_layer = 6

torch_dtype = torch.float32

output_folder = "eval_results/st"
artifacts_path = "artifacts"

force_rerun = True
save_activations = False
clean_up_activations = True


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


def main():
    print("[main] starting ST sparse probing")
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

    config = SparseProbingEvalConfig(model_name=model_name)
    config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[model_name]
    config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[model_name]

    # fast smoke-test config
    config.dataset_names = ["fancyzhx/ag_news"]
    config.probe_train_set_size = 200
    config.probe_test_set_size = 100
    config.k_values = [1]

    selected_saes = [("eleuther_pythia160m_st_k64_65k_layers6mlp_sparse_fast", sae)]

    print("[main] wrapper ready")
    print(f"[main] device={device}")
    print(f"[main] hook={tl_hook_name}")
    print(f"[main] output_folder={output_folder}")
    print(f"[main] dataset_names={config.dataset_names}")
    print(f"[main] probe_train_set_size={config.probe_train_set_size}")
    print(f"[main] probe_test_set_size={config.probe_test_set_size}")
    print(f"[main] k_values={config.k_values}")

    try:
        sparse_probing.run_eval(
            config=config,
            selected_saes=selected_saes,
            device=device,
            output_path=output_folder,
            force_rerun=force_rerun,
            clean_up_activations=clean_up_activations,
            save_activations=save_activations,
            artifacts_path=artifacts_path,
        )
    except Exception:
        print("[main] sparse_probing.run_eval raised an exception")
        traceback.print_exc()
        raise

    print("[main] done")
    print(f"[main] output folder contents: {os.listdir(output_folder)}")


if __name__ == "__main__":
    main()