# Demo: segment a particle, see it classified

A small Gradio page that joins the two halves of the paper end to end. You click
on a particle in a TEM micrograph, SAM proposes a mask from that click, and the
paper's classifier (frozen DINOv2 ViT-B/14 hypercolumns pooled **inside the
mask**, then the MLP head) predicts Fiber / Cluster / Matrix / MatrixSurface.

It is deliberately not the full analysis application (that is
[`sem_analysis_app/`](../sem_analysis_app/README.md)): no scale bars, no
measurement, no batch mode. Just click, mask, label.

## What is in here

| file | purpose |
|---|---|
| `demo_app.py` | the page: `python demo_app.py`, then open http://127.0.0.1:7861 |
| `classify.py` | `CNTClassifier`: one image + one mask -> label and probabilities. Uses the same `encoder_bench.build_encoder` / `pool_maps` code the benchmark used |
| `weights/dinov2_probe_head.pt` | the MLP head (8 MB, fp16) and the feature scaler |
| `train_demo_head.py` | regenerates the head from the feature cache (needs `CNT_BASE`) |
| `demo_images.csv` | the 60 micrographs the demo uses, and why they were chosen |
| `select_demo_images.py` | regenerates that list from the results files |
| `fetch_demo_images.py` | copies or downloads the 60 images into `images/` (not in git) |

## Setup

```bat
conda activate cnt-vfm                              :: environment.yml at the repo root
python download_sam_weights.py --model vit_b        :: run at the repo root; 358 MB, once
cd demo
python fetch_demo_images.py --download              :: 60 files, ~85 MB, from Harvard Dataverse
python demo_app.py
```

If you already have the dataset locally, set `CNT_BASE` and drop `--download`.
DINOv2 weights (330 MB) are fetched by `timm` from the Hugging Face hub on first
use. A GPU is not required; on CPU each click takes a few seconds.

## What the demo images are, and why they are easy

The 60 images are **held-out test images**: 20 Fiber, 20 Cluster, 20 Matrix,
chosen by `select_demo_images.py` as the test images that the largest number of
the 118 evaluated configurations classified correctly (58 of the 60 were
classified correctly by every configuration). They are the *unambiguous* cases
on purpose: the demo is meant to show the mechanism working, not to estimate
accuracy. The accuracy estimate is the paper's, on the whole test set.

MatrixSurface is omitted because its particles (fibres protruding from a large
carbonaceous particle) are the hardest to isolate with a single click, which
makes for a poor demonstration of the segmentation half.

## What the classifier is, exactly

`weights/dinov2_probe_head.pt` is **one** MLP head: the `test_refit` variant of
the pre-specified configuration, i.e. a single model trained on all 1,606
development images for the protocol's transferred epoch budget (11 epochs, seed
42) with early stopping off, produced by `train_demo_head.py` through the same
`probe_fit` code as the benchmark. Scored once on the 179-image test set it gets
**92.18 % (165/179)**, which is the `test_refit_acc` of that row in
[`results/results_probes.csv`](../results/results_probes.csv).

The paper's headline test number, 92.7 % (166/179), is the **five-fold
ensemble** of that configuration. Shipping the ensemble would mean five heads;
the single refit is within one image of it and is what the demo uses.

`train_demo_head.py` refuses to save a head whose predictions differ from the
protocol's own fit closure, so the shipped weights cannot silently be something
else.

## What to expect

Driving the page programmatically with **one positive click at the deepest
interior point of each ground-truth mask** (the click a person would make),
SAM's highest-scoring mask has a median IoU of 0.81 against the expert mask
and the classifier labels 51 of the 60 images correctly. Seven of the nine
misses are Cluster images where the click lands on one fibre of the bundle,
SAM returns that single fibre (IoU with the expert mask near zero), and the
classifier says Fiber, which is the right label for the mask it was given. A
second positive click on another part of the bundle grows the mask and the
label follows. That is the behaviour the demo is meant to make visible: the
label is a function of the mask, and the mask is a function of the clicks.

## Reading the output

Each click reruns SAM on all points so far (green = particle, red = background),
overlays the mask in the colour of the predicted class, and prints the four
class probabilities. The true class is shown next to the prediction for
comparison. The classifier only ever sees the pixels inside the mask, so a mask
that spills onto background, or covers only part of a fibre, changes the answer;
that sensitivity is the point of mask-guided pooling and is worth exploring.
