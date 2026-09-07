# Datasets and How We Prepare Them

A plain-language guide to the data behind this project, and the exact steps we apply
before any model sees it.

---

## 1. Two kinds of dataset

We use two different shapes of data, because we test two different kinds of constraint.

**(a) Hierarchies** — collections of series that must *add up*. These test **coherence**.

> **Example.** Australian tourism. If 12 visitors go to Sydney and 8 go to Melbourne,
> then "New South Wales + Victoria" must be 20, and "Australia" must equal the sum of
> all its states. That adding-up requirement is the **coherence constraint**.

> **The same shape without a second series.** A single monthly series must also add up
> *to itself* across time: its twelve monthly numbers have to sum to the annual one.
> That is **temporal coherence**, and we test it by taking flat M4-Monthly series and
> aggregating each one to bi-monthly, quarterly, four-monthly, semi-annual and annual
> totals (§4). It is the same adding-up constraint, so it reuses the summing-matrix
> machinery below — only now the "children" are consecutive months of one series rather
> than separate regions.

**(b) Flat collections** — thousands of unrelated individual series. These test
**non-negativity**, the **accuracy-vs-history** question, and (via time aggregation)
**temporal coherence**.

> **Example.** M5 retail: "units of product X sold at store Y each day". You cannot
> sell −3 items, so the forecast should never go below zero. That is the
> **non-negativity constraint**.

---

## 2. The eight hierarchies

These are the core of the paper. Five come from the standard Monash collection
(via the `datasetsforecast` package); the other three we build ourselves — a retail
hierarchy from M5, and two high-dependence energy hierarchies (Solar, Electricity) added
to populate the upper end of the cross-series dependence axis (§5.6).

| Hierarchy | What it is | Total nodes | Bottom | Aggregate | Horizon *h* | Season | Length |
|---|---|---:|---:|---:|---:|---:|---:|
| **TourismSmall** | Australian tourism, small grouping | 89 | 56 | 33 | 4 | 4 (quarterly) | 36 |
| **TourismLarge** | Australian tourism, deep grouping | 555 | 304 | 251 | 12 | 12 (monthly) | 228 |
| **Labour** | Australian labour-force counts | 57 | 32 | 25 | 8 | 12 | 503 |
| **Traffic** | Freeway sensor occupancy | 207 | 200 | 7 | 14 | 7 (daily) | 366 |
| **Wiki2** | Wikipedia page views | 199 | 150 | 49 | 14 | 7 | 366 |
| **M5 retail** *(we built this)* | Walmart sales, aggregated | 84 | 70 | 14 | 28 | 7 | 1969 |
| **Electricity** *(we built this)* | Consumption feeders, grouped | 46 | 40 | 6 | 24 | 24 (hourly) | 1081 |
| **Solar** *(we built this)* | PV stations, grouped | 152 | 137 | 15 | 24 | 24 (hourly) | 3000 |

**Reading the columns.**
- **Bottom** = the lowest-level series, the ones that get summed.
- **Aggregate** = every node *above* the bottom (totals and subtotals). These are the
  ones that can be *incoherent*, so they're what we measure.
- **Horizon _h_** = how many steps ahead we forecast.
- **Length** = how many time points each series has.

### The summing matrix `S`

Every hierarchy is encoded in one grid of 0s and 1s, `S`, which records **which small
things add up into which big things**. It is used as a matrix–vector product:

$$
y_{\text{all}} = S \, y_{\text{bottom}}
$$

- **each row** of `S` = one node of the hierarchy (a total, a subtotal, or a bottom series)
- **each column** = one bottom-level series
- **entry = 1** if that bottom series is part of that node, **0** otherwise

#### Worked example

Four bottom series — two regions × two product categories:

```
North/FOOD = 10     South/FOOD = 6
North/TOYS =  4     South/TOYS = 3
```

The hierarchy has 7 nodes (`Total`, `North`, `South`, and the 4 bottom), so `S` is
7 rows × 4 columns, and `y_bottom = [10, 4, 6, 3]`:

```
                N/FOOD  N/TOYS  S/FOOD  S/TOYS        row · y_bottom
        Total  [  1       1       1       1  ]  ->  10+4+6+3  = 23
        North  [  1       1       0       0  ]  ->  10+4      = 14
        South  [  0       0       1       1  ]  ->      6+3   =  9
       N/FOOD  [  1       0       0       0  ]  ->  10
       N/TOYS  [  0       1       0       0  ]  ->   4
       S/FOOD  [  0       0       1       0  ]  ->   6
       S/TOYS  [  0       0       0       1  ]  ->   3
```

**Read each row as a recipe.** The `North` row says "include N/FOOD and N/TOYS, ignore
the rest": `1×10 + 1×4 + 0×6 + 0×3 = 14`. The bottom rows are identity rows — each one
just picks out itself.

#### Why a matrix is worth the trouble

**1. Checking coherence becomes one comparison.** The model forecasts all 7 nodes
*independently*, producing its own `ŷ_all`. We compare that against what the bottom
level implies, `S × ŷ_bottom`. Suppose the model keeps the bottom values above but
predicts `North = 15` and `Total = 25`:

```
model says North = 15   but   S × ŷ_bottom = 14   ->  off by 6.7%   incoherent
model says Total = 25   but   S × ŷ_bottom = 23   ->  off by 8.7%   incoherent
```

Anything off by more than 1% counts as incoherent — that is the number reported in
our tables.

**2. Bottom-up reconciliation becomes one line.** Throw away the model's aggregate
forecasts and *recompute* them as `S × ŷ_bottom`. They now add up **by construction**:
this is not an approximation, the error is exactly zero.

**3. It generalises.** For multi-step forecasts `ŷ_bottom` becomes a
`bottom × horizon` matrix, and `S ×` it yields every node at every horizon in one
operation. The same object also handles "grouped" structures that are not strict trees
(e.g. slicing by region *and* by category at once).

---

## 3. How we built the M5 hierarchy

The five Monash hierarchies barely vary in *cross-series dependence* (how much the
bottom series move together), and that turned out to matter for our main result. So we
built a sixth from M5 retail to fill the gap.

**Starting point:** 30,490 series of the form "item *i* sold at store *j*, daily",
1,969 days each — about 60 million cells.

**Step 1 — aggregate to a manageable bottom level.** Individual items are too sparse
and too numerous. We sum all items within each *department* at each *store*:

```
FOODS_1_001 at CA_1  ┐
FOODS_1_002 at CA_1  ├─ sum ─>  bottom series "CA_1/FOODS_1"
FOODS_1_003 at CA_1  ┘
```

10 stores × 7 departments = **70 bottom series**.

**Step 2 — build the levels upward.** Store IDs already encode the state (`CA_1` is in
California), so the hierarchy falls out naturally:

```
                    Total                       (1 node)
                      |
        ┌─────────────┼─────────────┐
       CA            TX            WI           (3 states)
        |
   ┌────┼────┬────┐
  CA_1 CA_2 CA_3 CA_4  ...                      (10 stores)
   |
 ┌─┴─────────┬──────────┐
CA_1/FOODS_1  CA_1/HOBBIES_1  ...               (70 store x dept = bottom)
```

Total: 1 + 3 + 10 + 70 = **84 nodes**, of which **14 are aggregates**.

**Step 3 — verify.** We check that the *actual* data satisfies the constraint exactly:

$$
\max \left| S \, y_{\text{bottom}} - y_{\text{all}} \right| = 0.0
$$

If this weren't ~0, the summing matrix would be wrong. It is exactly 0.

> The 47.6M-row aggregation runs **once** and is cached to
> `_data/m5_hierarchy_bottom.parquet`, so later runs are fast.

---

## 3b. The two energy hierarchies (high dependence)

M5 fills the *middle* of the dependence range but nothing near the top. Two energy
datasets from the GluonTS collection are strongly cross-correlated at the bottom level,
so we build a hierarchy from each, exactly like M5 (group the raw series, build the
levels upward, verify `max|S·y_bottom − y_all| = 0`).

- **Solar** — 137 photovoltaic stations (hourly output). Stations are grouped into 14
  geographic groups, giving `Total → group → station` (137 bottom, 15 aggregates). The
  bottom series share a strong diurnal/weather signal, so mean bottom-series dependence
  is **0.70** — the highest in the study.
- **Electricity** — Portuguese consumption series aggregated into 40 feeders, grouped
  into 5 groups: `Total → group → feeder` (40 bottom, 6 aggregates). Correlated loads
  give dependence **0.63**. (The *raw* 370 clients are only ~0.27; grouping into feeders
  is what raises the dependence, so we group before building the tree.)

These two are also where we report the **trained global** baselines (LightGBM, TCN) most
prominently: if any hierarchy should reward a model that sees all series jointly, it is a
strongly co-moving energy one — yet those models are still incoherent (§4).

---

## 4. The flat datasets

| Dataset | What it is | Used for |
|---|---|---|
| **M3** (Monthly) | Classic competition series, values far from zero | non-negativity control |
| **M4** (Monthly, Quarterly) | Long competition series | accuracy vs. history length; **temporal coherence** (Monthly) |
| **M5** (daily sample) | Intermittent retail counts, many zeros | non-negativity + accuracy |

M3 and M5 are deliberately contrasted: M3 values sit far above zero, M5 values hover
*at* zero. A model only reveals a non-negativity problem in the second case.

For **temporal coherence** we take 400 M4-Monthly series and, for each, ask the model
for both the monthly forecast and the directly-forecast aggregate at every time scale
`k ∈ {2, 3, 4, 6, 12}` months. The `k` monthly numbers must sum to the aggregate; the
gap between the two is measured exactly as in §5.5, with the summing matrix now folding
consecutive months of one series instead of separate bottom series.

---

## 5. Preparation steps (applied identically to every model)

This is the part that makes the comparison fair.

### 5.1 Rolling-origin evaluation (not k-fold!)

Standard k-fold cross-validation is **wrong** for time series: random folds would let a
model train on the future to predict the past. The correct analogue is a **rolling
origin** — repeatedly cut the series at successively earlier points.

With horizon `h` and window index `w = 0, 1, 2, …`:

```
window 0:  [--------- train ---------][ test ]
window 1:  [------ train ------][ test ][ ... ]
window 2:  [--- train ---][ test ][ ...      ]
```

Formally, for a series of length $L$, horizon $h$ and window index $w$:

$$
\text{test}_w = y_{L-(w+1)h+1 \,:\, L-wh}, \qquad \text{train}_w = y_{1 \,:\, L-(w+1)h}
$$

We request **8** windows.

### 5.2 Adaptive window count

Short hierarchies cannot support 8 non-overlapping windows. TourismSmall has only 36
points with `h = 4`; asking for 8 windows would need `4 × 8 = 32` points of test alone,
leaving almost no training data. So we cap the count to what the *shortest* series
supports:

$$
n_{\text{windows}} = \min\left( n_{\text{requested}},\ \left\lfloor \frac{L_{\min} - 5}{h} \right\rfloor \right)
$$

In practice: TourismSmall gets **7** windows, every other hierarchy gets **8**. Capping
by the shortest series keeps the set of nodes identical across windows, which
reconciliation requires.

### 5.3 Scaling errors (MASE)

Raw errors aren't comparable across series — Wikipedia page views are in millions,
tourism counts in hundreds. We divide every error by a per-series scale, the average
change of a seasonal-naive forecast on the **full** series:

$$
\text{scale} = \frac{1}{L-m}\sum_{t=m+1}^{L} \left| y_t - y_{t-m} \right|,
\qquad \text{MASE} = \frac{\frac{1}{h}\sum_{t=1}^{h}\left| y_t - \hat{y}_t \right|}{\text{scale}}
$$

where $m$ is the seasonal period.

**MASE = 1** means "as good as seasonal naive". Below 1 is better. Because the scale is
computed once from the whole series, it stays constant across windows and history
lengths, so numbers remain comparable.

### 5.4 Turning quantiles into samples (for the probabilistic tests)

The models emit **quantiles** (e.g. the 10th, 50th, 90th percentile), but to reconcile
*distributions* we need **samples**. We convert by inverse-CDF sampling: draw a uniform
number `u ∈ [0,1]` and read off the quantile function at `u`.

We then combine the bottom-level samples in three ways:

1. **Independent** — just add the samples up, pretending series are unrelated.
2. **Schaake shuffle** — reorder samples to match the *historical* co-movement pattern.
3. **Gaussian copula** — impose a fitted correlation matrix instead.

Scored with **CRPS** (a proper scoring rule for distributions; lower is better).

### 5.5 Measuring incoherence

For each aggregate node we compare its own forecast against the sum of its children,
and call it incoherent if they differ by more than **1%** relative:

$$
\frac{\left| \hat{y}_{\text{node}} - \sum_{c \in \text{children}} \hat{y}_c \right|}{\left| \hat{y}_{\text{node}} \right|} > 0.01
$$

### 5.6 Measuring dependence

For the probabilistic result we need one number per hierarchy describing how much the
bottom series move together. We use the mean absolute correlation of the bottom series'
*first differences* (differencing removes shared trend, which would inflate it):

| Hierarchy | Dependence |
|---|---:|
| Wiki2 | 0.04 |
| TourismLarge | 0.11 |
| Labour | 0.19 |
| TourismSmall | 0.31 |
| **M5 retail** | **0.34** |
| Traffic | 0.44 |
| **Electricity** | **0.63** |
| **Solar** | **0.70** |

M5 was added to populate the middle of this range, and the two energy hierarchies
(Electricity, Solar) to populate the high end. That high end turned out to be decisive:
the independent-summation CRPS cost does **not** rise with dependence once Electricity and
Solar are included (Traffic is a lone high-cost outlier), so the paper reports the *copula
benefit* rather than a cost-scaling "law" (§4.3).

---
