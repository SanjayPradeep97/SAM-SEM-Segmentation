#!/usr/bin/env python
"""Verify the `cnt-vfm` environment can run the classification pipeline.

    python classification/check_env.py

Exits 0 when everything reproduce.bat needs is importable and the GPU (if
any) is genuinely usable; 1 otherwise.  It does more than import things: a
PyTorch wheel built for an older CUDA imports fine on an RTX 50-series card,
reports a device, and then falls back to CPU or dies on the first kernel, so
the check inspects the arch list and runs a real matmul.
"""
import importlib, os, sys, traceback

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "
problems, notes = [], []


def line(status, name, detail=""):
    print(f"  [{status}] {name:22s} {detail}")


def check_import(mod, required=True, label=None):
    label = label or mod
    try:
        m = importlib.import_module(mod)
        line(OK, label, getattr(m, "__version__", ""))
        return m
    except Exception as e:
        (problems if required else notes).append(f"{label}: {e}")
        line(BAD if required else WARN, label, str(e)[:60] if required else "not installed")
        return None


print("=" * 70); print("ENVIRONMENT CHECK  (classification pipeline)"); print("=" * 70)
print(f"  python  {sys.version.split()[0]}\n  exe     {sys.executable}")

print("\n-- required for reproduce.bat --")
torch = check_import("torch")
check_import("torchvision")
check_import("timm")
check_import("huggingface_hub")
check_import("sklearn", label="scikit-learn")
check_import("scipy")
check_import("numpy")
check_import("PIL", label="pillow")
check_import("matplotlib")
check_import("seaborn")
check_import("cv2", label="opencv-python")
check_import("segment_anything")

print("\n-- required for the fine-tuned YOLO baselines (stage 3 only) --")
check_import("ultralytics", required=False)

print("\n-- required for the Luo baseline and the demo --")
check_import("xgboost", required=False)
check_import("gradio", required=False)

print("\n-- data root --")
base = os.environ.get("CNT_BASE")
if not base:
    notes.append("CNT_BASE is not set (needed for anything that touches images or caches)")
    line(WARN, "CNT_BASE", "not set")
else:
    line(OK if os.path.isdir(os.path.join(base, "NIOSH Dataset")) else WARN, "CNT_BASE", base)
    if not os.path.isdir(os.path.join(base, "NIOSH Dataset")):
        notes.append(f"{base} has no 'NIOSH Dataset' directory")

if torch is not None:
    print("\n-- GPU --")
    try:
        avail = torch.cuda.is_available()
        line(OK if avail else WARN, "cuda.is_available", str(avail))
        if avail:
            cap = torch.cuda.get_device_capability(0)
            sm = f"sm_{cap[0]}{cap[1]}"
            line(OK, "device", torch.cuda.get_device_name(0))
            line(OK, "compute capability", sm)
            arches = torch.cuda.get_arch_list()
            if sm not in arches:
                problems.append(
                    f"this torch has no {sm} code (arch list: {' '.join(arches)}).  Reinstall:\n"
                    f"      pip install --force-reinstall torch torchvision "
                    f"--index-url https://download.pytorch.org/whl/cu128")
                line(BAD, "sm support", f"{sm} NOT in arch list")
            else:
                line(OK, "sm support", f"{sm} present")
            a = torch.randn(2048, 2048, device="cuda"); b = torch.randn(2048, 2048, device="cuda")
            c = (a @ b).sum().item(); torch.cuda.synchronize()
            line(OK, "matmul on GPU", f"ran, checksum {c:.3e}")
            free, total = torch.cuda.mem_get_info()
            line(OK, "VRAM", f"{free/1e9:.1f} GB free / {total/1e9:.1f} GB")
        else:
            notes.append("no CUDA device: everything runs on CPU, feature extraction ~10x slower")
    except Exception:
        problems.append("GPU check raised:\n" + traceback.format_exc(limit=3))
        line(BAD, "GPU check", "raised, see below")

print("\n" + "=" * 70)
if problems:
    print("NOT READY"); print("=" * 70)
    for p in problems: print(f"  * {p}")
    for n in notes: print(f"  - {n}")
    sys.exit(1)
print("READY"); print("=" * 70)
for n in notes: print(f"  note: {n}")
sys.exit(0)
