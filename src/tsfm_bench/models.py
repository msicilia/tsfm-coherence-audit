"""Model wrappers behind one interface.

Each model maps a history array -> a Forecast(median, q10, q90). Chronos-Bolt
runs batched on the GPU; classical models use statsforecast; CAA-real selects a
small classical model conditioned on declared constraints and projects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


QUANTILE_CROSSINGS = {"checked": 0, "repaired": 0}


def _check_quantiles(out, n_series: int, h: int, levels, model: str = "") -> np.ndarray:
    """Enforce the wrapper output contract: quantiles shaped (n_series, h, n_levels),
    non-decreasing across levels.

    Two failure modes, treated differently on purpose:

    * A wrong *shape* is a programming error; model APIs disagree about axis order
      (Chronos-2 emits (1, h, L), others (1, L, h)) and a silent transpose corrupts
      every downstream CRPS number. That raises.
    * *Quantile crossing* is a real, known model artefact (some models expose an
      explicit `fix_quantile_crossing` option). The standard repair is to sort along
      the level axis, so we do that and count it rather than aborting a study. The
      tolerance is relative, since an absolute epsilon is meaningless for series whose
      values are in the thousands.
    """
    out = np.asarray(out, dtype=float)
    want = (n_series, h, len(levels))
    if out.shape != want:
        raise ValueError(f"{model or 'model'}: quantile shape {out.shape} != {want}")

    QUANTILE_CROSSINGS["checked"] += 1
    scale = np.maximum(np.abs(out).max(), 1e-12)
    if np.any(np.diff(out, axis=2) < -1e-6 * scale):
        QUANTILE_CROSSINGS["repaired"] += 1
        out = np.sort(out, axis=2)
    return out


@dataclass
class Forecast:
    median: np.ndarray
    q10: np.ndarray
    q90: np.ndarray
    model: str = ""
    n_params: int = 0
    closed_form: bool = False
    rationale: str = ""


# --------------------------------------------------------------------------- #
# Chronos-Bolt (real TSFM, batched GPU inference)
# --------------------------------------------------------------------------- #
class ChronosBolt:
    name = "Chronos-Bolt"
    closed_form = False
    n_params = 48_000_000          # chronos-bolt-small, order of magnitude

    # rough param counts per checkpoint (order of magnitude, for the interpretability axis)
    _PARAMS = {"amazon/chronos-bolt-small": 48_000_000,
               "amazon/chronos-bolt-base": 205_000_000}

    def __init__(self, model_id: str = "amazon/chronos-bolt-small", device: str = "cuda",
                 name: str | None = None):
        import torch
        from chronos import BaseChronosPipeline

        self.torch = torch
        self.device = device
        self.pipe = BaseChronosPipeline.from_pretrained(
            model_id, device_map=device, dtype=torch.bfloat16)
        self.model_id = model_id
        if name is not None:
            self.name = name
        self.n_params = self._PARAMS.get(model_id, self.n_params)

    def predict_batch(self, histories: list[np.ndarray], h: int) -> list[Forecast]:
        """Forecast many series in one GPU call (ragged contexts allowed)."""
        torch = self.torch
        ctxs = [torch.tensor(np.asarray(y, dtype=np.float32)) for y in histories]
        q, _mean = self.pipe.predict_quantiles(
            ctxs, prediction_length=h, quantile_levels=[0.1, 0.5, 0.9])
        q = q.float().cpu().numpy()             # (n_series, h, 3)
        out = []
        for i in range(q.shape[0]):
            out.append(Forecast(
                median=q[i, :, 1], q10=q[i, :, 0], q90=q[i, :, 2],
                model=self.name, n_params=self.n_params, closed_form=False))
        return out

    def predict(self, history: np.ndarray, h: int) -> Forecast:
        return self.predict_batch([history], h)[0]

    def predict_quantiles_batch(self, histories, h, levels):
        """Return array (n_series, h, len(levels)) of predictive quantiles."""
        torch = self.torch
        ctxs = [torch.tensor(np.asarray(y, dtype=np.float32)) for y in histories]
        q, _ = self.pipe.predict_quantiles(ctxs, prediction_length=h,
                                           quantile_levels=list(levels))
        return _check_quantiles(q.float().cpu().numpy(), len(histories), h, levels, self.name)


# --------------------------------------------------------------------------- #
# Chronos-2 (Amazon): newer architecture, dedicated pipeline + quantile API
# --------------------------------------------------------------------------- #
class Chronos2:
    name = "Chronos-2"
    closed_form = False
    n_params = 120_000_000         # order of magnitude

    def __init__(self, device: str = "cuda"):
        import torch
        from chronos import Chronos2Pipeline
        self.torch = torch
        self.device = device
        self.pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)

    def predict_quantiles_batch(self, histories, h, levels):
        """predict_quantiles returns (list[Tensor], list[Tensor]); each quantile
        tensor is (n_variates=1, len(levels), h). -> (n_series, h, len(levels))."""
        torch = self.torch
        ctxs = [torch.tensor(np.asarray(y, dtype=np.float32)) for y in histories]
        qs, _means = self.pipe.predict_quantiles(
            ctxs, prediction_length=h, quantile_levels=list(levels))
        L = len(levels)
        out = np.empty((len(histories), h, L), dtype=float)
        for i, q in enumerate(qs):
            arr = np.squeeze(np.asarray(q.float().cpu().numpy()))   # (h, L) or (L, h)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape == (L, h) and L != h:             # (L,h) -> (h,L)
                arr = arr.T
            out[i] = arr                                   # (h, L)
        return _check_quantiles(out, len(histories), h, levels, self.name)

    def predict_batch(self, histories: list[np.ndarray], h: int) -> list[Forecast]:
        q = self.predict_quantiles_batch(histories, h, [0.1, 0.5, 0.9])
        return [Forecast(median=q[i, :, 1], q10=q[i, :, 0], q90=q[i, :, 2],
                         model=self.name, n_params=self.n_params)
                for i in range(len(histories))]

    def predict(self, history: np.ndarray, h: int) -> Forecast:
        return self.predict_batch([history], h)[0]


class Chronos2Joint(Chronos2):
    """Chronos-2 in its multivariate mode: every history passed to one call is treated
    as a variate of a single task, so group attention shares information across all of
    them (Ansari et al. 2025, Sec. 3). Used to test whether forecasting a whole
    hierarchy jointly yields coherent outputs. ``joint`` tells callers not to split the
    histories into batches."""
    name = "Chronos-2 (joint)"
    joint = True

    def predict_quantiles_batch(self, histories, h, levels):
        torch = self.torch
        n = len(histories)
        T = max(len(y) for y in histories)
        target = np.full((n, T), np.nan, dtype=np.float32)   # left-pad shorter variates
        for i, y in enumerate(histories):
            y = np.asarray(y, dtype=np.float32)
            target[i, T - len(y):] = y
        qs, _means = self.pipe.predict_quantiles(
            [torch.tensor(target)], prediction_length=h, quantile_levels=list(levels),
            batch_size=max(256, n))
        out = np.asarray(qs[0].float().cpu().numpy(), dtype=float)   # (n, h, L)
        if out.shape != (n, h, len(levels)):
            raise ValueError(f"{self.name}: unexpected quantile shape {out.shape}, "
                             f"wanted {(n, h, len(levels))}")
        return _check_quantiles(out, n, h, levels, self.name)


# --------------------------------------------------------------------------- #
# TimesFM 2.5 (Google): array API, batched
# --------------------------------------------------------------------------- #
class TimesFM:
    name = "TimesFM"
    closed_form = False
    n_params = 200_000_000

    def __init__(self, device: str = "cuda", max_context: int = 2048,
                 max_horizon: int = 64):
        import timesfm
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch")
        self.model.compile(timesfm.ForecastConfig(
            max_context=max_context, max_horizon=max_horizon,
            normalize_inputs=True, use_continuous_quantile_head=True,
            fix_quantile_crossing=True))

    def predict_batch(self, histories: list[np.ndarray], h: int) -> list[Forecast]:
        inputs = [np.asarray(x, dtype=np.float32) for x in histories]
        point, quant = self.model.forecast(horizon=h, inputs=inputs)
        point = np.asarray(point)
        quant = np.asarray(quant) if quant is not None else None
        out = []
        for i in range(len(inputs)):
            med = point[i][:h]
            if quant is not None and quant.ndim == 3 and quant.shape[2] >= 10:
                # quantile head returns deciles q0.1..q0.9 in cols 1..9 (col 0 = mean)
                q10, q90 = quant[i, :h, 1], quant[i, :h, 9]
            else:
                q10 = q90 = med
            out.append(Forecast(median=med, q10=q10, q90=q90,
                                model=self.name, n_params=self.n_params))
        return out

    def predict(self, history: np.ndarray, h: int) -> Forecast:
        return self.predict_batch([history], h)[0]

    def predict_quantiles_batch(self, histories, h, levels):
        """Return (n_series, h, len(levels)) quantiles. TimesFM emits deciles
        q0.1..q0.9 in columns 1..9; we interpolate to the requested levels."""
        inputs = [np.asarray(x, dtype=np.float32) for x in histories]
        _point, quant = self.model.forecast(horizon=h, inputs=inputs)
        quant = np.asarray(quant)                       # (n,h,10): col0=mean, 1..9 deciles
        deciles = np.arange(1, 10) / 10.0
        out = np.empty((len(inputs), h, len(levels)), dtype=float)
        for i in range(len(inputs)):
            for t in range(h):
                out[i, t] = np.interp(levels, deciles, quant[i, t, 1:10])
        return _check_quantiles(out, len(inputs), h, levels, self.name)


# --------------------------------------------------------------------------- #
# Moirai 1.1-R (Salesforce): gluonts sample-path predictor
# --------------------------------------------------------------------------- #
class Moirai:
    name = "Moirai"
    closed_form = False
    n_params = 91_000_000          # moirai-1.1-R-small, order of magnitude

    def __init__(self, size: str = "small", device: str = "cuda",
                 num_samples: int = 100):
        from uni2ts.model.moirai import MoiraiModule
        self.module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}")
        self.device = device
        self.num_samples = num_samples

    def predict_batch(self, histories: list[np.ndarray], h: int) -> list[Forecast]:
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai import MoiraiForecast

        ctx = max(len(x) for x in histories)
        model = MoiraiForecast(
            module=self.module, prediction_length=h, context_length=ctx,
            patch_size="auto", num_samples=self.num_samples,
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        predictor = model.create_predictor(
            batch_size=min(256, len(histories)), device=self.device)
        ds = ListDataset(
            [{"target": np.asarray(x, dtype=np.float32),
              "start": pd.Period("2000-01-01", freq="D")} for x in histories],
            freq="D")
        out = []
        for f in predictor.predict(ds):
            s = f.samples                          # (num_samples, h)
            out.append(Forecast(median=np.median(s, axis=0),
                                q10=np.quantile(s, 0.1, axis=0),
                                q90=np.quantile(s, 0.9, axis=0),
                                model=self.name, n_params=self.n_params))
        return out

    def predict(self, history: np.ndarray, h: int) -> Forecast:
        return self.predict_batch([history], h)[0]

    def predict_quantiles_batch(self, histories, h, levels):
        """Moirai is a sample-path predictor, so quantiles come directly from its
        sample ensemble. Returns (n_series, h, len(levels))."""
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai import MoiraiForecast

        ctx = max(len(x) for x in histories)
        model = MoiraiForecast(
            module=self.module, prediction_length=h, context_length=ctx,
            patch_size="auto", num_samples=self.num_samples,
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        predictor = model.create_predictor(
            batch_size=min(256, len(histories)), device=self.device)
        ds = ListDataset(
            [{"target": np.asarray(x, dtype=np.float32),
              "start": pd.Period("2000-01-01", freq="D")} for x in histories],
            freq="D")
        out = np.empty((len(histories), h, len(levels)), dtype=float)
        for i, f in enumerate(predictor.predict(ds)):
            out[i] = np.quantile(f.samples, list(levels), axis=0).T   # (h, L)
        return _check_quantiles(out, len(histories), h, levels, self.name)


# --------------------------------------------------------------------------- #
# TinyTimeMixer (IBM Granite): a *tiny* (<1M parameter) foundation model
# --------------------------------------------------------------------------- #
class TinyTimeMixer:
    """Granite TTM. Two structural constraints shape this wrapper:

    * fixed context (512) and horizon (96). Shorter histories are left-padded and
      the padding is flagged via `past_observed_mask`, so the model is told which
      positions are real rather than being fed fabricated history.
    * point forecasts only. It emits no predictive quantiles, so
      `predict_quantiles_batch` raises rather than returning a degenerate
      distribution that would silently corrupt CRPS.
    """

    name = "TinyTimeMixer"
    closed_form = False
    n_params = 805_280

    def __init__(self, device: str = "cuda",
                 model_id: str = "ibm-granite/granite-timeseries-ttm-r2"):
        import torch
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        self.torch = torch
        self.device = device
        self.model = TinyTimeMixerForPrediction.from_pretrained(model_id).to(device).eval()
        self.ctx = int(self.model.config.context_length)
        self.max_h = int(self.model.config.prediction_length)

    def _prepare(self, histories):
        """-> (past_values, past_observed_mask) padded/truncated to the fixed context."""
        torch = self.torch
        n = len(histories)
        x = np.zeros((n, self.ctx, 1), dtype=np.float32)
        mask = np.zeros((n, self.ctx, 1), dtype=np.float32)
        for i, y in enumerate(histories):
            y = np.asarray(y, dtype=np.float32)
            take = min(len(y), self.ctx)
            x[i, self.ctx - take:, 0] = y[-take:]        # right-aligned (most recent)
            mask[i, self.ctx - take:, 0] = 1.0
            if take < self.ctx:                          # left-pad with the earliest value
                x[i, :self.ctx - take, 0] = y[0]
        return (torch.tensor(x, device=self.device),
                torch.tensor(mask, device=self.device))

    def predict_batch(self, histories: list[np.ndarray], h: int) -> list[Forecast]:
        if h > self.max_h:
            raise ValueError(f"{self.name} supports horizons up to {self.max_h}, got {h}")
        torch = self.torch
        past, mask = self._prepare(histories)
        with torch.no_grad():
            out = self.model(past_values=past, past_observed_mask=mask)
        pred = out.prediction_outputs.float().cpu().numpy()[:, :h, 0]
        return [Forecast(median=pred[i], q10=pred[i], q90=pred[i],
                         model=self.name, n_params=self.n_params)
                for i in range(len(histories))]

    def predict(self, history: np.ndarray, h: int) -> Forecast:
        return self.predict_batch([history], h)[0]

    def predict_quantiles_batch(self, histories, h, levels):
        raise NotImplementedError(
            f"{self.name} emits point forecasts only; it is excluded from the "
            f"probabilistic-coherence study rather than given degenerate quantiles."
        )


# --------------------------------------------------------------------------- #
# Classical models (statsforecast)
# --------------------------------------------------------------------------- #
class Classical:
    """A single statsforecast model fit per series, with quantiles via levels."""

    def __init__(self, kind: str, period: int):
        from statsforecast.models import (AutoARIMA, AutoETS, AutoTheta,
                                           Naive, SeasonalNaive)
        self.kind = kind
        self.period = period
        self._params = {"SeasonalNaive": 1, "Naive": 1, "AutoTheta": 2,
                        "AutoETS": 10, "AutoARIMA": 5}.get(kind, 5)
        ctor = {"AutoARIMA": AutoARIMA, "AutoETS": AutoETS, "AutoTheta": AutoTheta,
                "SeasonalNaive": SeasonalNaive, "Naive": Naive}[kind]
        m = max(1, period)
        if kind in ("AutoARIMA", "AutoETS", "AutoTheta", "SeasonalNaive"):
            self.model = ctor(season_length=m)
        else:
            self.model = ctor()
        self.name = kind

    @property
    def n_params(self):
        return self._params

    closed_form = True

    def predict(self, history: np.ndarray, h: int) -> Forecast:
        y = np.asarray(history, dtype=float)
        try:
            self.model.fit(y)
            res = self.model.predict(h=h, level=[80])
            median = res["mean"]
            q10 = res.get("lo-80", median)
            q90 = res.get("hi-80", median)
        except Exception:                       # robustness: fall back to last value
            median = np.repeat(y[-1], h)
            q10 = q90 = median
        return Forecast(median=np.asarray(median, dtype=float),
                        q10=np.asarray(q10, dtype=float),
                        q90=np.asarray(q90, dtype=float),
                        model=self.name, n_params=self._params, closed_form=True)


# --------------------------------------------------------------------------- #
# CAA-real: constraint-aware selection over the small classical library + projection
# --------------------------------------------------------------------------- #
class CAA:
    name = "CAA"
    closed_form = True

    def __init__(self, period: int, non_negative: bool = True,
                 upper_bound: float | None = None,
                 constraint_aware_selection: bool = True):
        self.period = period
        self.non_negative = non_negative
        self.upper_bound = upper_bound
        # ablation switch: if False, model SELECTION ignores declared constraints
        # (feature-only), though the final projection still enforces feasibility.
        self.constraint_aware_selection = constraint_aware_selection

    def _diagnose(self, y: np.ndarray) -> dict:
        n = len(y)
        m = self.period
        t = np.arange(n)
        slope, intercept = np.polyfit(t, y, 1) if n >= 2 else (0.0, y[-1])
        fitted = slope * t + intercept
        trend = max(0.0, 1.0 - np.sum((y - fitted) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-9))
        if m > 1 and n >= 2 * m:
            sm = np.array([y[i::m].mean() for i in range(m)])
            sfull = np.array([sm[i % m] for i in range(n)])
            seas = max(0.0, 1.0 - np.sum((y - sfull) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-9))
        else:
            seas = 0.0
        return dict(n=n, trend=trend, seas=seas, intermit=float(np.mean(np.abs(y) < 1e-9)))

    def _choose(self, d: dict) -> str:
        reasons = []
        short = d["n"] < 96
        if d["intermit"] > 0.3:
            reasons.append(f"intermittency {d['intermit']:.2f}>0.3 -> SeasonalNaive")
            return "SeasonalNaive", reasons
        if (self.constraint_aware_selection and self.upper_bound is not None
                and (short or d["trend"] > 0.6)):
            reasons.append("capacity + (short or strong trend) -> AutoTheta (damped)")
            return "AutoTheta", reasons
        if d["seas"] > 0.5 and d["n"] >= 2 * self.period:
            reasons.append(f"seasonal {d['seas']:.2f}>0.5 -> AutoETS")
            return "AutoETS", reasons
        if short:
            reasons.append(f"short history n={d['n']}<96 -> AutoTheta (low-variance)")
            return "AutoTheta", reasons
        reasons.append("ample history -> AutoARIMA")
        return "AutoARIMA", reasons

    def _project(self, arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        if self.non_negative:
            out = np.maximum(out, 0.0)
        if self.upper_bound is not None:
            out = np.minimum(out, self.upper_bound)
        return out

    def predict(self, history: np.ndarray, h: int) -> Forecast:
        y = np.asarray(history, dtype=float)
        d = self._diagnose(y)
        kind, reasons = self._choose(d)
        base = Classical(kind, self.period).predict(y, h)
        median = self._project(base.median)
        q10 = self._project(base.q10)
        q90 = self._project(base.q90)
        rationale = (f"Chose {kind} [{'; '.join(reasons)}]; "
                     f"projected onto {'y>=0' if self.non_negative else ''}"
                     f"{f', y<={self.upper_bound:g}' if self.upper_bound else ''}.")
        return Forecast(median=median, q10=q10, q90=q90, model=self.name,
                        n_params=base.n_params, closed_form=True, rationale=rationale)
