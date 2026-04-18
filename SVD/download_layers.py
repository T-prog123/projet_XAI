from huggingface_hub import hf_hub_download
import os

def main():
    repo_id = "EleutherAI/skip-transcoder-Llama-3.2-1B-131k"
    
    # Let's just grab 3 layers to start: early (0), middle (8), and late (15)
    layers_to_download = range(16)

    print("Starting download...")
    for layer in layers_to_download:
        folder_name = f"layers.{layer}.mlp"
        filename = f"{folder_name}/sae.safetensors"
        
        print(f"Downloading {filename} (this is ~2.1 GB, it might take a moment)...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir="." # This automatically creates the folders in your current directory
            )
            print(f"Successfully downloaded {folder_name}!")
        except Exception as e:
            print(f"Error downloading {folder_name}: {e}")

    print("\nAll targeted layers downloaded. You can now run analyze_w_skip.py!")

if __name__ == "__main__":
    main()