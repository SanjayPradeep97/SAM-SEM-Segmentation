#!/usr/bin/env python3
"""
Download SAM Model Weights

Downloads the Segment Anything Model (SAM) weights from Meta AI's repository.
Weights are saved to the sam_weights/ directory.

Available models:
- ViT-H (vit_h): Best quality, largest model (2.4 GB)
- ViT-L (vit_l): Good balance of quality and speed (1.2 GB)
- ViT-B (vit_b): Fastest, smallest model (358 MB)
"""

import os
import urllib.request
from pathlib import Path

# Model download URLs
MODELS = {
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
        "size": "2.4 GB"
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
        "size": "1.2 GB"
    },
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "size": "358 MB"
    }
}


def download_file(url, destination):
    """Download file with progress bar."""
    print(f"Downloading to: {destination}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '=' * filled + '-' * (bar_length - filled)

        if total_size > 0:
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f'\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='', flush=True)

    urllib.request.urlretrieve(url, destination, progress_hook)
    print()  # New line after progress bar


def main():
    # Create sam_weights directory
    weights_dir = Path(__file__).parent / "sam_weights"
    weights_dir.mkdir(exist_ok=True)

    print("SAM Model Weights Downloader")
    print("=" * 60)
    print("\nAvailable models:")
    for key, model in MODELS.items():
        print(f"  {key}: {model['filename']} ({model['size']})")

    print("\nWhich model(s) would you like to download?")
    print("  1. ViT-H (best quality, recommended)")
    print("  2. ViT-L (good balance)")
    print("  3. ViT-B (fastest)")
    print("  4. All models")
    print("  5. ViT-H and ViT-B (recommended for most users)")

    choice = input("\nEnter choice (1-5): ").strip()

    models_to_download = []
    if choice == "1":
        models_to_download = ["vit_h"]
    elif choice == "2":
        models_to_download = ["vit_l"]
    elif choice == "3":
        models_to_download = ["vit_b"]
    elif choice == "4":
        models_to_download = ["vit_h", "vit_l", "vit_b"]
    elif choice == "5":
        models_to_download = ["vit_h", "vit_b"]
    else:
        print("Invalid choice. Exiting.")
        return

    print(f"\nDownloading {len(models_to_download)} model(s)...")

    for model_key in models_to_download:
        model = MODELS[model_key]
        destination = weights_dir / model["filename"]

        if destination.exists():
            print(f"\n✓ {model['filename']} already exists. Skipping.")
            continue

        print(f"\n📥 Downloading {model['filename']} ({model['size']})...")
        try:
            download_file(model["url"], destination)
            print(f"✓ Downloaded successfully!")
        except Exception as e:
            print(f"✗ Error downloading {model['filename']}: {e}")

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Weights saved to: {weights_dir.absolute()}")


if __name__ == "__main__":
    main()
