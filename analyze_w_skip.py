import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # -----------------------------
    # 1. Configuration & Setup
    # -----------------------------
    model_name = "HuggingFaceTB/SmolLM2-135M"
    w_skip_dir = "outputs/sst_small_demo/w_skips"
    
    # Identify which layers were saved
    saved_files = [f for f in os.listdir(w_skip_dir) if f.endswith('.pt')]
    saved_files.sort() # Ensure we process them in order
    
    if not saved_files:
        print(f"No .pt files found in {w_skip_dir}. Did the training script finish?")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading tokenizer and unembedding matrix from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gpt = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    W_unembed = gpt.lm_head.weight.detach().cpu().float()

    # Prepare the plot
    plt.figure(figsize=(10, 6))

    # -----------------------------
    # 2. Process Each Layer
    # -----------------------------
    for file_name in saved_files:
        layer_str = file_name.replace("W_skip_layer_", "").replace(".pt", "")
        file_path = os.path.join(w_skip_dir, file_name)
        
        print(f"\n========================================")
        print(f" ANALYZING LAYER {layer_str}")
        print(f"========================================")
        
        W_skip = torch.load(file_path).float()
        
        # --- Perform SVD ---
        U, S, Vh = torch.linalg.svd(W_skip)
        
        # --- Add to Plot ---
        plt.plot(S.numpy(), marker='.', linestyle='-', markersize=2, label=f"Layer {layer_str}")
        
        # --- Unembedding Trick (Logit Lens) ---
        top_k_vectors = 3  # Analyze top 3 directions per layer
        top_tokens = 8
        
        for i in range(top_k_vectors):
            print(f"\n  [Vector #{i+1}] Singular Value: {S[i].item():.2f}")
            
            input_dir = Vh[i, :] 
            output_dir = U[:, i] 
            
            # Project onto vocab
            in_logits = torch.matmul(W_unembed, input_dir)
            out_logits = torch.matmul(W_unembed, output_dir)
            
            # Get Top Excitatory Tokens
            in_tokens = [tokenizer.decode(idx.item()).strip() for idx in torch.topk(in_logits, top_tokens).indices]
            out_tokens = [tokenizer.decode(idx.item()).strip() for idx in torch.topk(out_logits, top_tokens).indices]
            
            # Get Top Inhibitory (Negative) Tokens for Input
            in_bot_tokens = [tokenizer.decode(idx.item()).strip() for idx in torch.topk(in_logits, top_tokens, largest=False).indices]
            
            print(f"    Read (+) : {in_tokens}")
            print(f"    Read (-) : {in_bot_tokens}")
            print(f"    Write (+): {out_tokens}")

    # -----------------------------
    # 3. Finalize Plot
    # -----------------------------
    plt.title("Singular Values of $W_{skip}$ Across Layers")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Magnitude (Log Scale)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(w_skip_dir, "w_skip_svd_comparison.png"))
    print(f"\nSaved cross-layer SVD plot to {os.path.join(w_skip_dir, 'w_skip_svd_comparison.png')}")

if __name__ == "__main__":
    with torch.no_grad():
        main()