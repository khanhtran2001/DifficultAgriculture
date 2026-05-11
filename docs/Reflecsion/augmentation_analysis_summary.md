# Augmentation Comparison Analysis: GWHD-2021 Dataset

## Summary Table: Baseline vs Augmented Performance

| Seed | Baseline COCO AP | Augmented COCO AP | Δ AP (aug - baseline) | Δ AP% |
|------|-----------------|-------------------|----------------------|-------|
| 123  | 0.2307          | 0.1756            | **-0.0551**          | -23.9% |
| 456  | 0.2127          | 0.2550            | **+0.0423**          | +19.9% |
| 789  | 0.2453          | 0.2412            | **-0.0041**          | -1.7%  |
| **Mean** | **0.2296**  | **0.2239**        | **-0.0056 ± 0.0398** | **-2.4%** |

---

## Detailed Metrics Breakdown

### COCO AP (Primary Metric)
| Metric | Baseline Mean | Augmented Mean | Δ Mean | Std Dev Δ |
|--------|--------------|----------------|--------|-----------|
| **AP** | 0.2296       | 0.2239         | -0.0056| ±0.0398   |
| **AP50** | 0.5736     | 0.5509         | -0.0227| ±0.0571   |
| **AP75** | 0.1417     | 0.1408         | -0.0009| ±0.0473   |

### Size-Stratified Performance
| Metric | Baseline Mean | Augmented Mean | Δ Mean | Interpretation |
|--------|--------------|----------------|--------|---|
| **AP_small** | 0.0603 | 0.0639 | **+0.0035** | Slight improvement for small objects |
| **AP_medium** | 0.2567 | 0.2523 | -0.0044 | Slight degradation for medium objects |
| **AP_large** | 0.3775 | 0.3530 | -0.0245 | Noticeable degradation for large objects |

---

## Key Analysis Insights

### 1. **High Seed-Level Variability**
The most striking finding is the **extreme inconsistency across seeds**:
- Seed 456: augmentation **helped** (+4.2 AP points)
- Seed 123: augmentation **hurt** (-5.5 AP points)  
- Seed 789: essentially **no change** (-0.4 AP points)

This ±0.0398 standard deviation (~174% of the mean effect) indicates that **augmentation success is highly stochastic** and not reliably reproducible on GWHD-2021.

### 2. **Net Effect: Marginal Degradation**
Across all seeds, augmented models show a **mean AP loss of -0.56 percentage points** (95% CI spans from -0.046 to +0.034, crossing zero). This suggests:
- Copy-paste augmentation does **not reliably improve detection** on GWHD-2021
- The effect is **statistically indistinguishable from zero** given the variability

### 3. **Size-Specific Patterns**
- **Small objects (+0.35 AP):** Modest improvement—augmentation does add diversity for small object detection
- **Medium objects (-0.44 AP):** Slight degradation, possibly due to increased occlusion or crowding
- **Large objects (-2.45 AP):** Most significant loss—augmented large objects may have unrealistic spatial distributions or boundary artifacts

### 4. **Confidence Metrics Reveal Detection Drift**
- **AP50** drops more than AP: -2.27 vs -0.56
- **AP75** nearly unchanged: -0.09

This pattern suggests augmentation introduces **more localization inaccuracy** (AP50 is more sensitive) than outright misses. Likely causes:
- Pasted objects may not align perfectly with backgrounds
- Occlusion patterns differ from natural data
- Network overfits to artificial spatial arrangements

### 5. **Comparison to MinneApple (from your paper)**
| Dataset | Baseline AP | Augmented AP | Δ AP | Variability |
|---------|-----------|-------------|------|-----------|
| MinneApple | 0.353 | 0.348 | -0.005 | ±0.005 (low) |
| GWHD-2021 | 0.230 | 0.224 | -0.006 | ±0.040 (very high) |

GWHD-2021 shows **8× higher variability**, suggesting:
- The dataset may have **less stable augmentation behavior** than MinneApple
- Domain characteristics may amplify sensitivity to augmentation hyperparameters
- Train-test data distribution may be less aligned

---

## Conclusions & Recommendations

### What Worked (Seed 456)
When augmentation **did help** (+4.2 AP), it likely benefited from:
- Favorable weight initialization
- Better alignment between augmented and held-out test distributions
- Possibly different gradient dynamics during training

### What Went Wrong (Seed 123)
The significant failure (-5.5 AP) suggests:
- Copy-paste augmentation can introduce **biased or unrealistic patterns**
- The network may have overfit to synthetic object placements
- Large object degradation hints at boundary/context misalignment

### Actionable Insights
1. **Augmentation is not a reliable default** on GWHD-2021—test each seed independently
2. **Avoid large object regions** in copy-paste; they degrade most
3. **Small object augmentation shows promise** (+0.35 AP)—consider size-stratified augmentation
4. **Run diagnostics** on training curves per seed to identify failure modes
5. **Consider hybrid strategies**: augment only small objects, or use selective copy-paste with domain adaptation

---

## Statistical Significance
- **Effect size:** -0.0056 AP (negligible to small)
- **95% CI:** [-0.0454, 0.0342] (crosses zero—not statistically significant)
- **Conclusion:** Copy-paste augmentation is **not recommended as a general strategy** for GWHD-2021 without further validation or dataset-specific tuning
