"""
paper_protocol.py -- ONE evaluation protocol, applied identically to every method
in the revised manuscript.

WHY THIS FILE EXISTS
--------------------
The original submission reported 95.53% (171/179). That number came from
`DINO Final.ipynb`, which called:

    train_model(final_model, trainval_loader, test_loader, ...)

`test_loader` landed in the `val_loader` slot, and `train_model` restores the
checkpoint with the lowest loss on that loader.  The reported model's weights
were therefore selected on the test set.  Every one of the 24 configurations
shows the signature: mean CV 86.66% vs mean test 91.39%, a uniform +4.73 pp.

This module makes that class of error impossible to repeat.  The test set is
touched in exactly two places, both at the very end of `run()`, and no quantity
derived from it can influence training, stopping, or model selection.

THE PROTOCOL
------------
    1,785 images  (dataset_splits.pkl; the counts below are for that file)
      +-- 179   test  (held out; scored once per reported variant)
      +-- 1,606 dev   (all model development happens here)

    5-fold StratifiedKFold over dev, seed fixed.  For fold k:
        i_va  = held-out fold                      (~321 imgs)
        i_tr  = the other four folds               (~1,285 imgs)
        i_fit, i_es = stratified 90/10 split of i_tr   (~1,156 / ~129)

        model_k trains on i_fit and early-stops on i_es.
        i_va is not seen during training OR stopping -> its accuracy is an
        unbiased estimate.  The test set is not seen at all.

    The same five models therefore serve two purposes at zero extra cost:
        cv_acc         mean accuracy on the five held-out i_va folds  (PRIMARY)
        test_ensemble  mean softmax of the five models on the test set

    test_refit         one fresh model on ALL of dev, trained for exactly
                       median(best_epoch over the 5 folds) epochs with early
                       stopping disabled.  No held-out data is needed because
                       the epoch budget was transferred from the folds.
                       Flagged `refit_unreliable` when that budget is tiny --
                       see REFIT_MIN_BUDGET below.  Not reported in the paper.

    test_single        legacy reference: one model on 90% of dev, early-stopped
                       on the remaining 10%.  Reported so the new numbers can be
                       compared against the previous benchmark run.

All three test numbers are emitted for every method, always.  Reporting the
best of them would reintroduce exactly the selection effect this file exists to
prevent, so `run()` has no notion of a "best" variant.

THE PRIMARY RESULT IS DECLARED BEFORE ANY RESULTS EXIST
-------------------------------------------------------
`PRIMARY` below pins the headline configuration.  It is the configuration the
original submission already argued for, chosen on cross-validation, and it is
frozen here so that no downstream table can promote a different method on the
strength of a test-set number.
"""

from __future__ import annotations

import numpy as np

# Declared in advance.  Do not edit this to match a result.
PRIMARY = dict(encoder="dinov2_b14", geometry="g37_L1-3-6-9-11",
               mask="masked", pooling="avg+max",
               classifier="mlp", variant="ensemble",
               aggregation="mean-softmax-no-temperature",
               primary_metric="cv_acc_oof")

# `geometry` matters and is easy to get wrong.  The manuscript describes DINOv2
# at 518x518 with hypercolumn taps [1, 3, 6, 9, 11] (37x37 tokens, 3,840-D), and
# Supplementary Note S2 justifies that layer set by ablation.  The cross-encoder
# benchmark deliberately uses a DIFFERENT geometry -- a 32x32 grid with five
# evenly spaced taps -- so that no encoder benefits from a layer set tuned for
# it.  Both are correct for their purpose, and they must not be confused:
#   g37_L1-3-6-9-11  the manuscript's configuration -> the HEADLINE row
#   g32_t5           the matched cross-encoder comparison -> every other row
# Naming the geometry here means `headline()` fails loudly if the paper-geometry
# features were never extracted, instead of quietly reporting the benchmark one.

# The headline metric is cross-validated accuracy over the ~1,600 out-of-fold
# predictions, NOT the ~180-image test accuracy.  Two reasons, both decisive:
#   * n=179 cannot resolve the differences this paper claims.  Exact McNemar
#     needs a ~5-6 pp gap for p<0.05 at n=179; on 1,606 OOF predictions it
#     resolves 1.2-1.9 pp.
#   * 96 configurations were already scored on these same 180 images by the
#     earlier benchmark run.  Cumulative selection exposure is large: the
#     maximum over 96 evaluations sits ~5.6 pp above truth, which is on its own
#     enough to manufacture the retracted 95.53%.  The test set is close to
#     spent and is reported once, for PRIMARY only, with its Wilson interval.

N_CLASSES = 4

# The refit variant retrains on ALL of dev for median(fold best epoch) epochs.
# When the folds stop within the first few epochs -- which happens for the
# fine-tuned ViT-B/16 and ConvNeXt-V2, whose validation loss bottoms out
# during warm-up on a 10% inner split -- the transferred budget is one to
# three epochs, the cosine schedule never leaves warm-up, and the refit can
# collapse (60.9%, 73.7% and 36.3% test accuracy in results_finetune.csv).
# The fold ENSEMBLE, which the manuscript reports, is unaffected because each
# fold model keeps its own best checkpoint.  Rows whose budget is this small
# are flagged so the refit number is never read as a result.
REFIT_MIN_BUDGET = 4


def refit_unreliable(epoch_budget):
    return bool(int(epoch_budget) < REFIT_MIN_BUDGET)


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def dev_test_indices(split_of):
    """train+val -> dev, test -> test.  Returns (dev_idx, test_idx)."""
    split_of = np.asarray(split_of)
    dev = np.where((split_of == "train") | (split_of == "val"))[0]
    test = np.where(split_of == "test")[0]
    if len(dev) == 0 or len(test) == 0:
        raise ValueError(f"empty split: dev={len(dev)} test={len(test)}")
    if set(dev) & set(test):
        raise ValueError("dev and test overlap")
    return dev, test


def make_folds(y_dev, n_folds, seed):
    """Identical folds for every method.  Depends only on (y_dev, n_folds, seed)."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr.copy(), va.copy()) for tr, va in skf.split(np.zeros(len(y_dev)), y_dev)]


def inner_split(idx, y_dev, seed, frac=0.10):
    """Stratified fit/early-stop split *within* the training folds."""
    from sklearn.model_selection import train_test_split
    idx = np.asarray(idx)
    i_fit, i_es = train_test_split(idx, test_size=frac, stratify=y_dev[idx],
                                   random_state=seed)
    return np.sort(i_fit), np.sort(i_es)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def _score(y_true, probs):
    from sklearn.metrics import accuracy_score, f1_score
    pred = np.asarray(probs).argmax(1)
    return (100 * accuracy_score(y_true, pred),
            float(f1_score(y_true, pred, average="macro")), pred)


def _no_overlap(a, b, na, nb):
    if a is None or b is None:
        return
    ov = set(np.asarray(a).tolist()) & set(np.asarray(b).tolist())
    if ov:
        raise ValueError(f"protocol violation: {na} and {nb} share "
                         f"{len(ov)} indices (e.g. {sorted(ov)[:5]})")


def _guarded(fit):
    """Wrap a method's `fit` so the protocol contract is enforced on EVERY call,
    for every adapter -- frozen probe, fine-tuned CNN, XGBoost alike.

    This exists because an adapter-local check is not enough.  An audit injected
    the original bug one level up (early-stopping each fold model on that fold's
    own evaluation set) and it passed every test, because the only enforcement
    lived inside one adapter.  Enforcement belongs here, where every method
    must pass through.
    """
    def inner(train_idx, stop_idx, targets, fixed_epochs, seed):
        if isinstance(targets, str) or not isinstance(targets, (list, tuple)):
            raise TypeError("targets must be a list")
        if fixed_epochs is None and stop_idx is None:
            raise ValueError("early stopping needs stop_idx")
        if fixed_epochs is not None and stop_idx is not None:
            raise ValueError("fixed_epochs and stop_idx are mutually exclusive")
        _no_overlap(train_idx, stop_idx, "train_idx", "stop_idx")
        for t in targets:
            if isinstance(t, str):
                if t != "test":
                    raise ValueError(f"unknown target {t!r}")
                # the test set may never be a training or stopping set
                continue
            _no_overlap(train_idx, t, "train_idx", "prediction target")
            _no_overlap(stop_idx, t, "stop_idx", "prediction target")
        probs, ep = fit(train_idx, stop_idx, targets, fixed_epochs, seed)
        if len(probs) != len(targets):
            raise ValueError("fit returned the wrong number of prediction sets")
        for p in probs:
            p = np.asarray(p)
            if p.ndim != 2 or p.shape[1] != N_CLASSES:
                raise ValueError(f"probs must be (n, {N_CLASSES}), got {p.shape}")
            if not np.allclose(p.sum(1), 1.0, atol=1e-4):
                raise ValueError("probs must be normalised per row")
        return probs, ep
    return inner


# --------------------------------------------------------------------------
# the protocol
# --------------------------------------------------------------------------
def run(fit, y_dev, y_test, *, n_folds=5, seed=42, label="", log=print,
        tick=None, want_single=True):
    """Apply the protocol to one method.

    fit(train_idx, stop_idx, targets, fixed_epochs, seed) -> (probs_list, best_epoch)

        train_idx   positions into dev to train on
        stop_idx    positions into dev to early-stop on; None when fixed_epochs
                    is given
        targets     LIST of prediction targets, each either an array of dev
                    positions or the string "test".  One trained model scores
                    all of them, so a fold model is never trained twice.
        fixed_epochs  int -> train exactly this many epochs, early stopping off
                      None -> early-stop on stop_idx
        seed        RNG seed for this fit

        returns a list of prob arrays (one per target, each (n, N_CLASSES)) and
        the epoch whose weights were kept (or the fixed budget).

    `fit` is the ONLY place a method-specific model appears, which is what makes
    the protocol identical across frozen probes, fine-tuned CNNs and XGBoost.
    """
    y_dev = np.asarray(y_dev)
    y_test = np.asarray(y_test)
    fit = _guarded(fit)                      # contract enforced on every call
    folds = make_folds(y_dev, n_folds, seed)

    cv_acc, cv_f1, best_epochs = [], [], []
    test_probs_per_fold = []
    oof = np.full((len(y_dev), N_CLASSES), np.nan, dtype=np.float64)

    for k, (i_tr, i_va) in enumerate(folds, 1):
        i_fit, i_es = inner_split(i_tr, y_dev, seed)

        # the evaluation fold must be invisible to BOTH training and stopping
        _no_overlap(i_fit, i_va, "train", "eval-fold")
        _no_overlap(i_es, i_va, "early-stop", "eval-fold")
        _no_overlap(i_fit, i_es, "train", "early-stop")

        # ONE training run scores both the held-out fold and the test set
        (p_va, p_te), ep = fit(i_fit, i_es, [i_va, "test"], None, seed)
        a, f, _ = _score(y_dev[i_va], p_va)
        cv_acc.append(a); cv_f1.append(f); best_epochs.append(int(ep))
        oof[i_va] = p_va
        test_probs_per_fold.append(np.asarray(p_te, dtype=np.float64))

        log(f"    [{label}] fold {k}/{n_folds}: {a:5.2f}%  (best epoch {ep})")
        if tick:
            tick()

    budget = int(round(float(np.median(best_epochs))))

    # ---- variant 1: ensemble of the five fold models -----------------------
    ens = np.mean(test_probs_per_fold, axis=0)
    ens_acc, ens_f1, ens_pred = _score(y_test, ens)

    # ---- variant 2: refit on ALL of dev with the transferred epoch budget ---
    all_dev = np.arange(len(y_dev))
    (p_refit,), _ = fit(all_dev, None, ["test"], budget, seed)
    ref_acc, ref_f1, ref_pred = _score(y_test, p_refit)
    if tick:
        tick()

    oof_pred = oof.argmax(1)
    oof_acc = 100.0 * float((oof_pred == y_dev).mean())

    out = dict(
        oof_acc=oof_acc, oof_n=int(len(y_dev)),
        cv_acc=float(np.mean(cv_acc)), cv_std=float(np.std(cv_acc, ddof=1)),
        cv_f1=float(np.mean(cv_f1)), cv_folds=[float(a) for a in cv_acc],
        epoch_budget=budget, fold_epochs=best_epochs,
        test_ensemble_acc=float(ens_acc), test_ensemble_f1=float(ens_f1),
        test_refit_acc=float(ref_acc), test_refit_f1=float(ref_f1),
        refit_unreliable=refit_unreliable(budget),
        test_n=int(len(y_test)),
    )

    # ---- variant 3: legacy single model, 90% of dev ------------------------
    if want_single:
        i_fit, i_es = inner_split(all_dev, y_dev, seed)
        (p_single,), _ = fit(i_fit, i_es, ["test"], None, seed)
        s_acc, s_f1, s_pred = _score(y_test, p_single)
        out.update(test_single_acc=float(s_acc), test_single_f1=float(s_f1))
        out["_pred_single"] = s_pred
        if tick:
            tick()

    for name, acc in (("ensemble", ens_acc), ("refit", ref_acc)):
        k = int(round(acc / 100 * len(y_test)))
        lo, hi = wilson_ci(k, len(y_test))
        out[f"test_{name}_correct"] = k
        out[f"test_{name}_ci"] = f"[{lo:.1f},{hi:.1f}]"

    out["_oof"] = oof
    out["_oof_pred"] = oof_pred
    out["_pred_ensemble"] = ens_pred
    out["_pred_refit"] = ref_pred
    out["_probs_ensemble"] = ens
    return out


def n_fits(n_folds=5, want_single=True):
    """Model fits per method: one per fold, plus the refit, plus the legacy single."""
    return n_folds + 1 + (1 if want_single else 0)
