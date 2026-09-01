# Luo et al. (2021) baseline, re-implemented on the paper's split

`luo_baseline.py` re-implements the VGG-16 hypercolumn + VLAD + gradient-boosting
classifier of Luo et al., *Nanoscale Adv.* 3, 206–213 (2021), and evaluates it on
exactly the images, folds and held-out test set every other method in the paper
uses. It is the "Luo et al. (re-implementation)" row of Table 3.

## Files

| file | purpose |
|---|---|
| `luo_baseline.py` | the re-implementation: features (VGG-16 hypercolumns, K-means dictionary, VLAD) and the XGBoost classifier |
| `run_all.py` | runs every descriptor normalisation x {clean, leaky} and writes `results/luo_baseline_summary.csv` |
| `environment.yml` | minimal conda env (`cnt-luo`); the pipeline env `../environment.yml` also works |
| `results/luo_results_*.json` | one file per run: configuration, CV folds, held-out test result, confusion matrix |
| `results/luo_baseline_summary.csv` | all runs in one table (rebuild with `python run_all.py --report`) |

`../classification/luo_results.py` is what `make_tables.py` and
`verify_manuscript.py` use to pick the reported row from these files.

## What is reproduced from the paper

* greyscale TEM images normalised to 224x224; frozen ImageNet VGG-16
* hypercolumns from `b1c2, b2c2, b3c2, b4c3, b5c3` = 64+128+256+512+512 =
  **1472 dims/pixel**, matching the paper's stated 1472
* hypercolumn density 10 % (5,018 descriptors per image; the paper's "over 73
  million features" for a full-density hypercolumn is 50,176 x 1472)
* K-means dictionary with **K = 50**; VLAD residual encoding, 50 x 1472 =
  73,600-D per image
* gradient-boosted softmax classifier

## Deliberate deviations, all logged at runtime

1. **XGBoost** (histogram, GPU when available) instead of the unspecified 2020
   gradient-boosting implementation.
2. **Descriptor normalisation** is not specified by Luo et al. Three options are
   implemented (`none`, `l2`, `blockl2`) and **all three are run**; the paper
   reports the one with the best *cross-validated* accuracy under the clean
   protocol. The held-out test column is never used to choose. See the table
   below.
3. VLAD post-processing is signed square root + global L2 (standard practice;
   `--no-vlad-postnorm` disables it).
4. **No augmentation.** Luo et al. augmented minority classes to correct an
   8 % / 32 % / 24 % imbalance. The split used here has 430–462 images per class
   (a 7.4 % spread), so the step is omitted.

## Protocol

Splits come from `../splits/dataset_splits.pkl`, the same file every other
method uses. Primary metric: 5-fold `StratifiedKFold(shuffle=True,
random_state=42)` over the 1,606 development images. Secondary: one evaluation
on the 179 held-out test images.

```
--earlystop clean   (default)  an inner 10 % split of the training data picks
                               the number of boosting rounds; the model is then
                               refit on 100 % of the training data for that many
                               rounds.  The scored set never influences the fit.
--earlystop leaky              the set being scored is also the early-stopping
                               monitor.  This is the defect of the withdrawn
                               original analysis, reproduced ONLY to measure its
                               size.  Never a reported number.
```

**One documented caveat.** The K-means dictionary is fitted once on the `train`
rows, so the CV folds are scored under a dictionary that has seen most of their
images. It is unsupervised and label-free, and any bias runs *in favour of* the
baseline, but the CV figure is therefore not fully nested. The held-out test
figure is clean: no test image enters the dictionary or the fit.

## Results on `dataset_splits.pkl` (clean protocol)

See `results/luo_baseline_summary.csv` for every run, including the leaky
rows. The reported row is the clean run with the best CV accuracy; run
`python ../classification/luo_results.py` to print the selection.

## Running it

```bat
conda activate cnt-vfm            :: or cnt-luo from baseline/environment.yml
set CNT_BASE=<data root>          :: see ../classification/cnt_paths.py
cd baseline
python run_all.py                 :: none + l2 + blockl2, clean + leaky; ~1.5 h on an RTX 5080
python run_all.py --norms l2      :: one normalisation, clean + leaky; ~30 min
python run_all.py --report        :: rebuild the summary CSV from results/*.json
```

VLAD feature matrices (~0.5 GB per normalisation) are cached under
`<CNT_BASE>/Luo Baseline/`; the cache key covers every option that affects the
features and a mismatched cache is refused rather than reused. Runs are
bit-reproducible given the same GPU class.

## Correctness notes

* `grid_sample` sampling is numerically equivalent to bilinear-upsampling each
  feature map to 224x224 and indexing the chosen pixel (max abs error 1e-13
  over all pixels and all five map resolutions); `align_corners=False` and
  `padding_mode="border"` are both load-bearing.
* VLAD matches a literal per-descriptor reference implementation to 1e-7
  relative; residual accumulation runs in fp32.
* No train/test leakage in the clean path: the dictionary is fitted on train
  only, early-stopping sets never intersect scored sets, and
  `QuantileDMatrix(ref=...)` reuses the training sketch so binning cannot leak.
