# What We're Doing, and What We Found

Plain-language summary. No formulas. See `DATASETS.md` for the data details and
`paper/main.pdf` for the formal write-up.

---

## The one-paragraph version

Time-series "foundation models" are big pretrained networks that forecast without
being fitted to your specific data. They work well — but on smooth data they work about
*equally* well, and about as well as a ten-parameter classical model, so "which model
is most accurate" turns out to be a weak way to tell them apart. The thing that *does*
separate them, and separates all of them from a proper forecasting system, is a blind
spot: because each forecasts **each series on its own**, none can respect a rule that
ties its outputs together — whether that ties several series ("the regions must add up
to the national total") or ties one series to itself across time scales ("the twelve
monthly numbers must add up to the annual one"). We tested six of these models on six
real datasets and found this failure is near-total and has nothing to do with accuracy
or model size. Fixing it for ordinary forecasts is cheap. Fixing it for *uncertainty
estimates* is not, and we show why.

---

## The problem, by analogy

Imagine you hire four excellent forecasters and give each one a single store to
predict. Each is superb at their own store. But nobody is asked to make the store
predictions add up to the regional total — so they don't. Not because anyone is bad at
their job, but because **nobody was ever given the job of making them consistent**.

That is exactly what a per-series foundation model does. It's a structural property of
the setup, not a flaw you can train away. And it isn't only about separate stores: ask
one forecaster for both the monthly and the annual figure of the *same* store and
nobody made those agree either — same gap, no second store to blame (Finding 1b).

**Why it matters practically:** if you show a regional manager numbers that don't sum
to the national figure, the forecast is unusable no matter how accurate each piece is.
Budgets, supply plans, and headcount all rely on the parts adding up.

---

## What we actually did

1. Took **six foundation models** — TinyTimeMixer, Chronos-Bolt, Chronos-Bolt-Base,
   Chronos-2, Moirai, and TimesFM, spanning 0.8M to 200M parameters — plus a tiny
   classical model (AutoETS, ~10 parameters). (TinyTimeMixer produces no uncertainty
   ranges, so the uncertainty experiments use the other five.)
2. Ran them on **eight hierarchies** (tourism, labour, traffic, Wikipedia, a retail
   hierarchy we built from Walmart M5 data, and two strongly-correlated energy
   hierarchies — Solar and Electricity — we built to test the high-dependence end).
   We also added **two trained global models** (a LightGBM and a TCN), fit jointly
   across all series, to check the adding-up failure isn't just a per-series quirk.
3. Measured three things: do forecasts **add up**? do they stay **non-negative**? and
   how does accuracy change with the amount of **history** available?
4. Repeated everything over **8 rolling time windows** so results aren't a fluke of one
   date, with confidence intervals throughout.

---

## Finding 1 — The adding-up failure is near-total, and accuracy doesn't help

On six of the eight hierarchies, **most aggregate nodes were incoherent** (69–100% for
every foundation model, TinyTimeMixer aside) — two exceptions are noted below.

Three things make this hard to explain away:

- **Being better doesn't help.** Chronos-2, the newest and often the most accurate
  model, was 74–100% incoherent on the six affected hierarchies. Consistency was never
  part of what the model is doing.
- **Being bigger doesn't help either.** The models span 0.8M to 200M parameters — a
  factor of 250 — with no trend. The smallest (TinyTimeMixer) is incoherent at much the
  same rates as models 250× larger.
- **Training jointly doesn't help.** The two global models, which see all series at
  once, are incoherent too (LightGBM 50–100%, TCN 21–100%). So this isn't a per-series
  quirk — it's a property of *any* model that doesn't force its outputs to add up.

The small classical model failed the same way, which is the point: this is a property
of forecasting without a coupling constraint, not of neural networks or model size.

> One value bucks the trend: TinyTimeMixer on the retail hierarchy (21%) and Solar
> (28%). It is also the least accurate model in the study — a flat enough forecast is
> trivially closer to consistent without being useful.

> The two exceptions were Labour (3–21%) and Electricity, where the series follow one
> smooth shared trend, so independent forecasts happen to nearly line up by accident.

**The fix is cheap.** "Bottom-up reconciliation" — just add up the lowest level and use
those sums — makes everything add up *exactly*, and costs almost no accuracy. On
Traffic, Chronos-2 went from 1.267 to 1.269 (essentially free).

**Practical rule:** never deploy raw per-series foundation-model forecasts on a
hierarchy. Add the reconciliation step, applied to whichever model is most accurate.

---

## Finding 1b — The same failure with a *single* series (across time)

A sceptic can answer Finding 1 with "then model the whole hierarchy jointly." So we
removed the second series entirely. Take one monthly series and ask the same model for
both its monthly forecast and its annual forecast. The twelve monthly numbers must sum
to the annual one. They don't: every model contradicts *itself* on **37–81%** of these
checks, and the disagreement grows the further apart the
two views are (monthly vs annual is worst).

There is no "other series" to blame here — the model is simply inconsistent with its
own predictions at a different zoom level. And this time reconciliation isn't just
cheap, it's **strictly better**: summing the monthly forecasts beat the direct annual
forecast in *every one* of the 30 model-by-level tests. A model asked directly for an
annual number is both self-contradictory and less accurate than adding up its own
monthly numbers.

This is the strongest form of the argument: the failure is about *linear coupling of
any kind*, not about hierarchies of separate series.

---

## Finding 2 — Uncertainty is a different story (the interesting one)

Modern models don't just give one number; they give a **range** ("sales will be 100,
probably between 80 and 130"). Making those *ranges* add up is much harder than making
the single numbers add up.

Here's the intuition. Suppose two stores each "might be up 10 or down 10".

- If they move **independently**, the total is usually near zero — one goes up while
  the other goes down.
- If they move **together** (a holiday lifts both), the total swings by ±20.

The model tells you each store's range but **never tells you whether they move
together**. That information simply isn't there — a per-series model produces the
pieces, never the joint picture.

**What we first thought — and then corrected.** With six hierarchies it looked like the
penalty for naively adding the ranges grew *in proportion to how much the series move
together* (p = 0.015). That reading rested on two mistakes. It was carried by a single
dataset (Traffic); and it counted each hierarchy once per model, which overstates the
evidence, because the five models sharing a hierarchy all see the same dependence and are
not independent replicates. Counted properly — one point per hierarchy — those original
six already gave p = 0.27. When we then added the two genuinely high-dependence
hierarchies (Solar and Electricity) to test it, what was left **vanished** (slope +0.11,
p = 0.58): on both of those strongly-correlated hierarchies, naive summation was
essentially *free*. So there is no reliable "cost grows with dependence" law — Traffic is
just a lone expensive case, and we report this as a corrected negative result.

**What does hold — and it's the real point:** if you borrow the co-movement pattern from
historical data (a "Schaake shuffle" or a Gaussian copula), the uncertainty forecast gets
**better**, and it improves *most where the series are most dependent* (p = 0.005,
counted per hierarchy; it survives the same correction that killed the cost law). On
Solar, the coupled version beats even the raw forecast. That's the genuine finding: a
per-series model omits the joint picture, and supplying it from history pays off exactly
where co-movement is strongest — even though it can only be approximated, so exact
probabilistic coherence stays an open problem.

---

## Finding 3 — Which model "wins" depends entirely on which model you picked

We compared all five foundation models against a ten-parameter classical model
(AutoETS) across six history lengths.

On smooth monthly data the winner **kept changing** as history grew: AutoETS won
at the shortest history, Chronos-2 in the middle, TimesFM at the longest. And one
foundation model (Moirai) was **worse than the ten-parameter model at every single
history length**.

The sharpest way to say it: at the longest history the five foundation models ranged
from 0.924 to 1.194 in error, and the little classical model (0.993) sat **right in the
middle of that pack**. So the gap *between* foundation models is bigger than the gap
between them and the small model — "is a foundation model better?" has no answer until
you say *which one*.

The exception is intermittent retail data, where all five foundation models beat the
classical model cleanly and agree with each other. There the advantage is real and
robust. So the honest verdict is regime-dependent, not universal.

We verified this properly (paired Wilcoxon tests with false-discovery correction across
all 120 comparisons; 104 survive).

**But here's the sharper version.** Those tests ask "averaged over thousands of series,
is there a difference?" — and with that many series, even a tiny average gap shows up.
If you instead ask the question a practitioner actually cares about — *"for my series,
is one of these better?"* — a Diebold–Mariano test says: on smooth monthly/quarterly
data, the ten-parameter model and the foundation models are **statistically
indistinguishable on about 81% of individual series** (over 90% at some history
lengths). On intermittent retail data the foundation models genuinely win, on roughly
38–49% of series.

So on smooth data the choice mostly doesn't matter at the level where forecasts get
used. This is why we lead with coherence: coherence holds no matter which model you
pick, in both the cross-series and across-time directions. Accuracy comparisons don't.

## Finding 4 — Non-negativity is a per-model quirk, not a universal flaw

You cannot sell −3 items, so a sales forecast should never go below zero. On
intermittent retail data (lots of zeros), **four of the five models did**:

| Model | median below zero | lower bound below zero |
|---|---:|---:|
| Moirai | 48.4% | 91.4% |
| Chronos-Bolt | 31.7% | 93.4% |
| Chronos-Bolt-Base | 30.5% | 90.6% |
| Chronos-2 | 9.6% | 87.6% |
| **TimesFM** | **0.0%** | **0.0%** |

TimesFM is clean because it applies a built-in positivity rule. The others don't.

Two things worth noting:

- **It only happens near zero.** On data sitting far above zero (M3), every model is
  essentially clean. So it's a property of the near-zero regime, not a general defect.
- **Newer didn't mean fixed.** Chronos-2 cut its negative *medians* more than
  threefold versus Chronos-Bolt (31.7% → 9.6%) but barely improved its negative
  *lower bounds* (93.4% → 87.6%). The headline number improved; the uncertainty band
  did not. You still can't deploy it on non-negative data without clipping.

Unlike coherence, this one is a fixable deployment detail rather than a structural
limit — but it is not fixed for you by default.

---

## Finding 5 — Our own method: an honest negative result

We built **CAA** (constraint-aware assembly): pick a small interpretable model, then
force its output to obey the declared rules.

We ran an ablation to find out which half does the work — the *smart selection* or the
*forcing*. Result:

- Making the selection constraint-aware changed the chosen model on **27% of series**…
- …but produced **no accuracy benefit whatsoever** (p = 0.97).
- Meanwhile, removing the forcing step left **9.9% of forecasts violating** the limit.

**All the value is in the projection step, none in the clever selection.** We rewrote
the paper's claims to say exactly that, and removed the method from the title. It's a
negative result about our own contribution, and reporting it is cheaper than having a
reviewer find it.

---

## The bottom line

| Question | Answer |
|---|---|
| Do foundation-model forecasts add up across a hierarchy? | **No** — 62–100% of aggregates fail, on 5 of 6 datasets |
| Does a single series at least agree with itself across time? | **No** — models self-contradict on 37–81% of aggregates |
| Does a better model fix it? | **No** — the most accurate model fails just as much |
| Does a *bigger* model fix it? | **No** — 0.8M to 200M parameters, no trend |
| Is it expensive to fix? | **No** — for ordinary forecasts it's essentially free |
| Is it fixed for uncertainty ranges too? | **No** — this is the genuinely open problem |
| Which foundation model should I use? | **Depends on your data** — the ranking flips |
| Do they keep sales forecasts non-negative? | **Some do** (TimesFM), some don't (Chronos, Moirai) |

**If you take one thing away:** a foundation model is not a drop-in replacement for a
forecasting *system*. It gives you excellent individual predictions and no guarantee
that they are mutually consistent. Add the consistency step yourself — it's cheap for
point forecasts, and still an open research problem for uncertainty.
