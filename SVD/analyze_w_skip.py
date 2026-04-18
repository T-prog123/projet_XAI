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

    # Dictionary to store singular values for our enhanced plotting
    all_singular_values = {}

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
            
            # Save singular values to our dictionary for Step 4
            layer_num = int(folder.split('.')[1])
            all_singular_values[layer_num] = S.numpy()
            
            # --- Logit Lens ---
            top_k_vectors = 5  
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
    # 4. ENHANCED PLOTTING (Multi-Panel)
    # ==========================================
    if not all_singular_values:
        print("No valid data was found to plot.")
        return

    # We use a 1x2 subplot: one for the full view, one for the 'Head' zoom
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Use a sequential colormap to show depth progression
    cmap = plt.get_cmap('viridis', 16)

    # We iterate over sorted keys to ensure the legend order is correct (0 to 15)
    for i in sorted(all_singular_values.keys()):
        layer_s = all_singular_values[i]
        color = cmap(i)
        
        # Plot 1: Full Spectrum (Log Scale)
        ax1.plot(layer_s, color=color, alpha=0.7, label=f"L{i}")
        
        # Plot 2: Head Zoom (First 150 indices)
        # This is where the 'functional rank' is most visible
        ax2.plot(layer_s[:150], color=color, alpha=0.8, label=f"L{i}")

    # Format Full Plot
    ax1.set_title("Full W_skip Singular Value Spectrum", fontsize=14)
    ax1.set_yscale('log')
    ax1.set_xlabel("Singular Value Index")
    ax1.set_ylabel("Magnitude (Log Scale)")
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Format Zoomed Plot
    ax2.set_title("Zoomed View: The 'Functional' Head (0-150)", fontsize=14)
    ax2.set_xlabel("Singular Value Index")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True, alpha=0.3)

    # Unified Legend (Placed outside the plots on the right)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.98, 0.95), loc='upper right', title="Layer")
    
    plt.tight_layout()
    # Adjust the right margin so the layout doesn't overlap the outside legend
    plt.subplots_adjust(right=0.90)
    
    plt.savefig("w_skip_svd_enhanced.png", dpi=300)
    print("\nEnhanced plot saved to 'w_skip_svd_enhanced.png'")

if __name__ == "__main__":
    with torch.no_grad():
        main()