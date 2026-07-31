import os
import hashlib

from huggingface_hub import snapshot_download

from src.config import IronConfig

# ----------------------------
# Configuration
# ----------------------------

# Mapping of local model names to Hugging Face repository IDs
MODELS = {
    "vjepa2_vitl": "facebook/vjepa2-vitl-fpc64-256",
    "cotracker3": "facebook/cotracker3",
    "depth_anything_v2_small": "depth-anything/Depth-Anything-V2-Small-hf",
}

# Directory where model weights will be stored.
#
# Resolved against the repository root, not the current working directory. The
# previous value was the literal "../models/weights", which lands inside the
# repo only when the script is run from scripts/ and writes to a SIBLING of the
# repository when run from the root — the documented location. Weights would
# appear to download successfully and then be invisible to everything that
# looks them up through the config.
_CONFIG = IronConfig.load()
SAVE_DIR = str(_CONFIG.paths.resolved_models_dir / "weights")

# File to store SHA256 hashes for downloaded weights. These are the
# source-checkpoint provenance an export manifest records.
HASH_FILE = os.path.join(SAVE_DIR, "hashes.txt")

os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# SHA256 Function
# ----------------------------


def sha256sum(filepath):
    """
    Compute the SHA256 hash of a file.

    Args:
        filepath (str): Path to the file.

    Returns:
        str: Hexadecimal SHA256 digest.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------
# Download + Verify
# ----------------------------


def main():
    """
    Download model weights from Hugging Face and compute SHA256 hashes
    for verification.

    The script:
        1. Downloads selected model weight files
        2. Stores them in SAVE_DIR
        3. Computes SHA256 hashes for each weight file
        4. Logs hashes to HASH_FILE for integrity checking
    """
    print("Starting model download...\n")

    with open(HASH_FILE, "w") as hash_log:
        for name, repo_id in MODELS.items():
            print(f"Downloading {name} from {repo_id}...")
            local_path = snapshot_download(
                repo_id=repo_id,
                local_dir=os.path.join(SAVE_DIR, name),
                allow_patterns=["*.bin", "*.pt", "*.pth", "*.safetensors"],
                ignore_patterns=["*.msgpack", "*.h5"],
                # token=True FORCED authentication and failed on a machine with
                # no cached Hugging Face login, even though every repository
                # here is public. None uses a cached token when one exists and
                # falls back to anonymous access when it does not.
                token=None,
            )

            print(f"Saved to: {local_path}")

            # Iterate through downloaded files and compute hashes
            for root, _, files in os.walk(local_path):
                for file in files:
                    if file.endswith((".bin", ".pt", ".pth", ".safetensors")):
                        full_path = os.path.join(root, file)
                        digest = sha256sum(full_path)
                        hash_log.write(f"{name} | {file} | {digest}\n")
                        print(f"SHA256 ({file}): {digest}")

            print("-" * 50)

    print("\nAll models downloaded and hashes recorded.")


if __name__ == "__main__":
    main()
