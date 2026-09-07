"""Trained multi-series (global) baselines: LightGBM and a native-PyTorch TCN.

These exist to separate two things the per-series baselines cannot:
  * generality of the coherence failure. A global model still forecasts each node
    with an unconstrained head, so its aggregate forecast does not equal the sum of
    its children; it is incoherent out of the box, exactly like a per-series model.
  * accuracy as a weak, regime-dependent discriminator. On some regimes a trained
    global model wins.

These are *trained* models, unlike the zero-shot foundation models: they add a
``fit(histories, h)`` step to the wrapper interface. Zero-shot models no-op ``fit``;
the orchestrator calls it (when present) before ``predict_batch``. Both train once on
all node histories of a hierarchy and then forecast each node, the natural way a
practitioner would deploy a global model on a hierarchy.

LightGBM wraps Nixtla's ``mlforecast`` rather than a hand-rolled training loop, which
keeps lag construction and the train/predict split library-managed. The TCN is a compact
dilated causal-convolution network in native torch (no darts dependency, which would
fight the pinned Moirai/torch stack).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import Forecast


# --------------------------------------------------------------------------- #
# Global LightGBM (mlforecast)
# --------------------------------------------------------------------------- #
class GlobalLightGBM:
    """A single LightGBM trained across all series, recursive multi-step forecast.

    Seasonality is captured by seasonal lags {m, 2m, 3m} plus a rolling mean, rather
    than calendar dummies, so the model is robust across integer-indexed hierarchies
    with different seasonal periods (hourly energy, weekly retail, monthly tourism).
    """

    name = "LightGBM"
    closed_form = False
    n_params = 0                      # trained; a boosted-tree "param count" is not comparable

    def __init__(self, period: int, n_estimators: int = 300,
                 learning_rate: float = 0.03, seed: int = 42):
        self.period = max(1, int(period))
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.seed = seed
        self.fcst = None
        self._uids: list[str] = []

    def _lags(self, min_len: int) -> list[int]:
        m = self.period
        cand = sorted({1, 2, 3, m, 2 * m, 3 * m})
        lags = [l for l in cand if l < min_len - 1]
        return lags or [1]

    def fit(self, histories: list[np.ndarray], h: int):
        import lightgbm as lgb
        from mlforecast import MLForecast
        from mlforecast.lag_transforms import RollingMean

        self._uids = [f"n{i}" for i in range(len(histories))]
        min_len = min(len(x) for x in histories)
        lags = self._lags(min_len)
        m = self.period

        frames = []
        for uid, x in zip(self._uids, histories):
            x = np.asarray(x, dtype=float)
            frames.append(pd.DataFrame({"unique_id": uid, "ds": np.arange(len(x)), "y": x}))
        df = pd.concat(frames, ignore_index=True)

        lag_tf = {1: [RollingMean(window_size=m)]} if (1 < m < min_len) else {}
        self.fcst = MLForecast(
            models={self.name: lgb.LGBMRegressor(
                n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                random_state=self.seed, verbosity=-1, n_jobs=-1)},
            freq=1, lags=lags, lag_transforms=lag_tf)
        self.fcst.fit(df, static_features=[])
        return self

    def predict_batch(self, histories: list[np.ndarray], h: int) -> list[Forecast]:
        if self.fcst is None or len(histories) != len(self._uids):
            self.fit(histories, h)
        preds = self.fcst.predict(h)
        by = {uid: g.sort_values("ds")[self.name].to_numpy(dtype=float)[:h]
              for uid, g in preds.groupby("unique_id")}
        out = []
        for uid in self._uids:
            p = by.get(uid, np.zeros(h))
            if len(p) < h:                                   # short-series guard
                pad = p[-1] if len(p) else 0.0
                p = np.pad(p, (0, h - len(p)), constant_values=pad)
            out.append(Forecast(median=p, q10=p, q90=p, model=self.name))
        return out

    def predict(self, history: np.ndarray, h: int) -> Forecast:
        self.fit([history], h)
        return self.predict_batch([history], h)[0]


# --------------------------------------------------------------------------- #
# Global TCN (native PyTorch dilated causal convolutions)
# --------------------------------------------------------------------------- #
class GlobalTCN:
    """Dilated causal TCN trained across all series with a direct h-step head.

    Windows are normalised per input window (subtract window mean, divide by window
    std) so a single network generalises across the wildly different scales of a
    hierarchy's nodes (a national total vs one bottom series). Predictions are
    de-normalised with the same window statistics.
    """

    name = "TCN"
    closed_form = False

    def __init__(self, period: int, channels: int = 32, layers: int = 4,
                 kernel: int = 3, epochs: int = 40, lr: float = 1e-3,
                 batch_size: int = 256, device: str = "cuda", seed: int = 42):
        self.period = max(1, int(period))
        self.channels, self.layers, self.kernel = channels, layers, kernel
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.device = device
        self.seed = seed
        self.net = None
        self.L = None
        self.n_params = 0

    def _build_net(self, h: int):
        import torch
        import torch.nn as nn

        k, layers, ch = self.kernel, self.layers, self.channels

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                convs = []
                c_in = 1
                for i in range(layers):
                    d = 2 ** i
                    pad = (k - 1) * d
                    convs.append(nn.Conv1d(c_in, ch, k, dilation=d, padding=pad))
                    convs.append(nn.ReLU())
                    self.__dict__.setdefault("_pads", []).append(pad)
                    c_in = ch
                self.convs = nn.ModuleList(convs)
                self.pads = [(k - 1) * (2 ** i) for i in range(layers)]
                self.head = nn.Linear(ch, h)

            def forward(self, x):                     # x: (B, 1, L)
                z = x
                ci = 0
                for i in range(layers):
                    conv = self.convs[ci]; relu = self.convs[ci + 1]; ci += 2
                    z = conv(z)
                    if self.pads[i] > 0:
                        z = z[:, :, :-self.pads[i]]    # causal chomp
                    z = relu(z)
                return self.head(z[:, :, -1])          # (B, h) from last timestep

        return Net()

    def fit(self, histories: list[np.ndarray], h: int):
        import torch

        torch.manual_seed(self.seed)
        dev = self.device if torch.cuda.is_available() else "cpu"
        m = self.period
        min_len = min(len(x) for x in histories)
        self.L = int(min(max(2 * m, 32), max(8, min_len - h - 1)))

        # build normalised (input L, target h) windows from every series
        X, Yt = [], []
        for x in histories:
            x = np.asarray(x, dtype=float)
            if len(x) < self.L + h:
                continue
            for s in range(0, len(x) - self.L - h + 1):
                win = x[s:s + self.L]
                tgt = x[s + self.L:s + self.L + h]
                mu, sd = win.mean(), win.std() + 1e-8
                X.append((win - mu) / sd)
                Yt.append((tgt - mu) / sd)
        if not X:                                        # series too short to window
            self.net = None
            return self
        X = torch.tensor(np.asarray(X), dtype=torch.float32, device=dev).unsqueeze(1)
        Yt = torch.tensor(np.asarray(Yt), dtype=torch.float32, device=dev)

        self.net = self._build_net(h).to(dev)
        self.n_params = sum(p.numel() for p in self.net.parameters())
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        n = X.shape[0]
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(X[idx]), Yt[idx])
                loss.backward()
                opt.step()
        self.net.eval()
        self._dev = dev
        return self

    def predict_batch(self, histories: list[np.ndarray], h: int) -> list[Forecast]:
        import torch

        if self.net is None:
            self.fit(histories, h)
        if self.net is None:                             # still unfittable -> last value
            return [Forecast(median=np.repeat(np.asarray(x, float)[-1], h),
                             q10=np.repeat(np.asarray(x, float)[-1], h),
                             q90=np.repeat(np.asarray(x, float)[-1], h), model=self.name)
                    for x in histories]
        dev = self._dev
        xs, stats = [], []
        for x in histories:
            x = np.asarray(x, dtype=float)
            win = x[-self.L:] if len(x) >= self.L else np.pad(x, (self.L - len(x), 0),
                                                              constant_values=x[0])
            mu, sd = win.mean(), win.std() + 1e-8
            xs.append((win - mu) / sd); stats.append((mu, sd))
        xt = torch.tensor(np.asarray(xs), dtype=torch.float32, device=dev).unsqueeze(1)
        with torch.no_grad():
            pred = self.net(xt).cpu().numpy()            # (n, h) normalised
        out = []
        for i, (mu, sd) in enumerate(stats):
            p = pred[i] * sd + mu
            out.append(Forecast(median=p, q10=p, q90=p, model=self.name,
                                n_params=self.n_params))
        return out

    def predict(self, history: np.ndarray, h: int) -> Forecast:
        self.fit([history], h)
        return self.predict_batch([history], h)[0]
