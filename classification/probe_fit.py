"""
probe_fit.py -- the frozen-encoder adapter for paper_protocol.run().

The encoder is frozen, so features are extracted once and cached; only the
probe (linear or MLP head) is trained.  This file supplies the `fit` callable
the protocol expects and does two things carefully:

  * the StandardScaler is fit on the training indices ONLY, never on the
    early-stopping split, the evaluation fold, or the test set;
  * when `fixed_epochs` is given the early-stopping branch is skipped entirely,
    so the refit variant cannot consult held-out data by accident.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def build_head(d, kind, seed):
    """Deterministic given `seed`: torch.manual_seed is set by the caller
    immediately before construction, so default init is reproducible."""
    torch.manual_seed(seed)
    if kind == "linear":
        return nn.Linear(d, 4)
    return nn.Sequential(
        nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 4))


def make_probe_fit(X_dev, y_dev, X_test, kind, device,
                   epochs=100, patience=10, batch=64, lr=1e-3, wd=1e-4):
    """Return a `fit` closure matching the paper_protocol contract."""
    from sklearn.preprocessing import StandardScaler

    X_dev = np.ascontiguousarray(X_dev, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    y_dev = np.asarray(y_dev)

    def fit(train_idx, stop_idx, targets, fixed_epochs, seed):
        train_idx = np.asarray(train_idx)

        # --- scaler sees training rows only -------------------------------
        sc = StandardScaler().fit(X_dev[train_idx])

        Xtr = torch.tensor(sc.transform(X_dev[train_idx]), device=device)
        ytr = torch.tensor(y_dev[train_idx], device=device, dtype=torch.long)

        model = build_head(X_dev.shape[1], kind, seed).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        crit = nn.CrossEntropyLoss()

        use_es = fixed_epochs is None
        if use_es:
            if stop_idx is None:
                raise ValueError("stop_idx required when fixed_epochs is None")
            stop_idx = np.asarray(stop_idx)
            if set(train_idx.tolist()) & set(stop_idx.tolist()):
                raise ValueError("train_idx and stop_idx overlap")
            Xes = torch.tensor(sc.transform(X_dev[stop_idx]), device=device)
            yes = torch.tensor(y_dev[stop_idx], device=device, dtype=torch.long)

        n = len(train_idx)
        g = torch.Generator().manual_seed(seed)
        n_ep = epochs if use_es else int(fixed_epochs)
        best, wait, best_state, best_ep = float("inf"), 0, None, n_ep

        for ep in range(1, n_ep + 1):
            model.train()
            perm = torch.randperm(n, generator=g).to(device)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                if len(idx) < 2:          # BatchNorm needs >1 row
                    continue
                opt.zero_grad()
                crit(model(Xtr[idx]), ytr[idx]).backward()
                opt.step()
            if use_es:
                model.eval()
                with torch.no_grad():
                    vl = crit(model(Xes), yes).item()
                if vl < best - 1e-5:
                    best, wait, best_ep = vl, 0, ep
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        break
        if use_es and best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        out = []
        with torch.no_grad():
            for t in targets:
                Xt = X_test if isinstance(t, str) and t == "test" else X_dev[np.asarray(t)]
                z = torch.tensor(sc.transform(Xt), device=device)
                out.append(torch.softmax(model(z), dim=1).cpu().numpy())
        return out, best_ep

    return fit
