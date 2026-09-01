"""SAM ViT-B/16 patch-token activation map, matched to the DINOv2 recipe.

The DINOv2 maps in the submitted Figure 6 are the L2 norm of the patch tokens at
a given transformer block, min-max normalised to [0, 1] and shown with 'jet'.
The same quantity is computed here for SAM's image encoder so the two panels
measure the same thing.

SAM's image encoder has no CLS token, so the CLS-similarity map used for
DINOv2's layer 11 has no SAM counterpart -- which is exactly why the comparison
is made on activation norm, which both architectures define.

Preprocessing follows the original figure code: the micrograph is first resized
to 224x224 (as in the submitted figure), then to the encoder's native input.
"""
import numpy as np, torch, cv2
from segment_anything.modeling.image_encoder import ImageEncoderViT
from functools import partial
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # classification/
import cnt_paths


def sam_checkpoint():
    """sam_vit_b_01ec64.pth: SAM_CHECKPOINT, else <repo>/sam_weights/ (written by
    download_sam_weights.py --model vit_b), else <CNT_BASE>/."""
    cands = [os.environ.get("SAM_CHECKPOINT"),
             HERE.parent.parent / "sam_weights" / "sam_vit_b_01ec64.pth"]
    if cnt_paths.base(required=False) is not None:
        cands.append(cnt_paths.base() / "sam_vit_b_01ec64.pth")
    for c in cands:
        if c and Path(c).is_file():
            return str(c)
    raise SystemExit("SAM ViT-B checkpoint not found.  Run  python download_sam_weights.py "
                     "--model vit_b  at the repository root, or set SAM_CHECKPOINT.")


CKPT = None
PIXEL_MEAN = np.array([123.675, 116.28, 103.53])
PIXEL_STD = np.array([58.395, 57.12, 57.375])


def build_sam_encoder(device="cpu"):
    enc = ImageEncoderViT(
        depth=12, embed_dim=768, img_size=1024, mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=12,
        patch_size=16, qkv_bias=True, use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11], window_size=14, out_chans=256)
    sd = torch.load(CKPT or sam_checkpoint(), map_location="cpu")
    enc_sd = {k[len("image_encoder."):]: v for k, v in sd.items()
              if k.startswith("image_encoder.")}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=True), None
    enc.to(device).eval()
    return enc


@torch.no_grad()
def sam_activation_maps(image_rgb, layers=(1, 6, 11), device="cpu"):
    """Returns {layer: (64, 64) map in [0,1]} -- L2 norm of that block's tokens."""
    enc = build_sam_encoder(device)
    acts = {}

    def hook(i):
        def f(_m, _inp, out):
            acts[i] = out.detach()          # (B, H, W, C) for SAM blocks
        return f

    handles = [enc.blocks[i].register_forward_hook(hook(i)) for i in layers]

    img = cv2.resize(image_rgb, (224, 224))          # match the submitted figure
    img = cv2.resize(img, (1024, 1024))
    x = (img.astype(np.float64) - PIXEL_MEAN) / PIXEL_STD
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)
    enc(t)
    for h in handles:
        h.remove()

    out = {}
    for i, a in acts.items():
        g = torch.norm(a[0], dim=-1).cpu().numpy()    # (H, W)
        out[i] = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return out


if __name__ == "__main__":
    from PIL import Image
    p = cnt_paths.image_path("Fiber", "CNT-Fiber-0044.tif")
    im = np.asarray(Image.open(p).convert("RGB"))
    print("image", im.shape)
    m = sam_activation_maps(im)
    for k, v in sorted(m.items()):
        print(f"  block {k:2d}: map {v.shape}  min {v.min():.3f} max {v.max():.3f} "
              f"mean {v.mean():.3f}")
