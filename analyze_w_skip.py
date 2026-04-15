import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

def main():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    model_name = "meta-llama/Llama-3.2-1B"
    device = torch.device("cpu")
    
    # Process all 16 layers
    layer_folders = [f"layers.{i}.mlp" for i in range(16)]
    
    # Setup Log File
    log_file_path = "svd_analysis_log.txt"

    # ==========================================
    # 2. INITIALIZATION
    # ==========================================
    print(f"Loading tokenizer and unembedding matrix from {model_name} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gpt = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": device})
    W_unembed = gpt.lm_head.weight.detach().float()

    plt.figure(figsize=(10, 6))

    # Open log file for writing
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"SVD Analysis Log for {model_name}\n")
        log_file.write("="*50 + "\n\n")

        # ==========================================
        # 3. PROCESS EACH LAYER
        # ==========================================
        for folder in layer_folders:
            if not os.path.exists(folder) or not os.path.exists(os.path.join(folder, "sae.safetensors")):
                print(f"Skipping {folder} (not found).")
                continue
                
            file_path = os.path.join(folder, "sae.safetensors")
            print(f"Analyzing {folder}...")
            
            # --- Extract W_skip ---
            W_skip = None
            with safe_open(file_path, framework="pt", device="cpu") as f:
                if "W_skip" in f.keys():
                    W_skip = f.get_tensor("W_skip").float()
                elif "skip.weight" in f.keys():
                    W_skip = f.get_tensor("skip.weight").float()
                    
            if W_skip is None:
                continue

            # --- SVD ---
            U, S, Vh = torch.linalg.svd(W_skip)
            
            # Plot
            layer_num = folder.split('.')[1]
            plt.plot(S.numpy(), marker='.', linestyle='-', markersize=2, label=f"Layer {layer_num}")
            
            # --- Logit Lens ---
            top_k_vectors = 5  # Increased to get more data
            top_tokens = 10    
            
            log_file.write(f"=== LAYER {layer_num} ===\n")
            
            for i in range(top_k_vectors):
                log_file.write(f"  [Vector #{i+1}] Singular Value: {S[i].item():.2f}\n")
                
                input_dir = Vh[i, :] 
                output_dir = U[:, i] 
                
                in_logits = torch.matmul(W_unembed, input_dir)
                out_logits = torch.matmul(W_unembed, output_dir)
                
                in_tokens = [tokenizer.decode(idx.item()).strip() for idx in torch.topk(in_logits, top_tokens).indices]
                out_tokens = [tokenizer.decode(idx.item()).strip() for idx in torch.topk(out_logits, top_tokens).indices]
                in_bot_tokens = [tokenizer.decode(idx.item()).strip() for idx in torch.topk(in_logits, top_tokens, largest=False).indices]
                
                log_file.write(f"    Read (+) : {in_tokens}\n")
                log_file.write(f"    Read (-) : {in_bot_tokens}\n")
                log_file.write(f"    Write (+): {out_tokens}\n\n")

    # ==========================================
    # 4. FINALIZE PLOT
    # ==========================================
    plt.title("Singular Values of W_skip Across Layers")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Magnitude (Log Scale)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig("w_skip_svd_comparison.png", dpi=300)
    print(f"\nDone! Results saved to '{log_file_path}' and 'w_skip_svd_comparison.png'")

if __name__ == "__main__":
    with torch.no_grad():
        main()