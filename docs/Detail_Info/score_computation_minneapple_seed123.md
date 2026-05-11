# MinneApple Scoring Details (Seed 123)

## Source and context

This note documents how the scoring values are computed in this project, with real numbers taken from:

- score file: `/home/khanh/Projects/DifficultyAgri/.cache_result/no_trad_aug/minneapple/scoring/seed_123/score_results.json`
- scoring implementation: `dagri/scoring/min_scorer.py`
- scoring wrapper: `dagri/scoring/scorer.py`
- experiment driver: `experiments/02_only_scoring.py`
- config: `configs/experiments/minneapple_yolo.yaml`

---

## 1) What scores exist

There are two levels of score:

1. **Object score**: `S_obj`
2. **Image score**: `S_img`

From current outputs (seed 123):

- `scoring_weight_mode`: `balance_correlation`
- `selected_object_weight (w1)`: `0.9`
- `selected_false_positive_weight (w2)`: `0.9405517378930963`
- number of scored images: `536`
- number of scored objects: `22815`

---

## 2) Step-by-step scoring pipeline

### Step 1: Generate predictions at two confidence regimes

In `experiments/02_only_scoring.py`, the model predicts on train images twice:

- **Low confidence predictions** (`LOW_CONF_THRESHOLD = 0.0001`): used for object matching and object-level difficulty.
- **Optimal confidence predictions** (`optimal_conf_threshold` found from validation): used for false-positive rate and missed-detection rate.

Both prediction sets are saved and then passed to `Scorer.score(...)`.

### Step 2: Compute object difficulty `S_obj`

For each GT object in an image (`MinScorer.score`):

1. Read GT box and class from train labels.
2. Loop through low-confidence predictions of the same class.
3. Keep predictions where IoU with GT is at least `iou_threshold` (from config, `0.5`).
4. For each valid prediction, compute:

\[
\text{cost} = \alpha(1 - \text{conf}) + \beta(1 - \text{IoU})
\]

with config values:

- `alpha = 0.5`
- `beta = 0.5`

5. Set object score as minimum valid cost:

\[
S_{obj} = \min(\text{cost over valid predictions})
\]

6. If no prediction matches IoU threshold, assign maximum difficulty:

\[
S_{obj} = \alpha + \beta = 1.0
\]

Interpretation:

- High confidence and high IoU -> low object difficulty.
- Missed or poor-matching objects -> high object difficulty.

### Step 3: Compute per-image object component

For each image:

\[
\text{avg\_object\_score} = \frac{1}{N}\sum_{i=1}^{N} S_{obj,i}
\]

where `N` is number of GT objects in the image.

### Step 4: Compute image-level false-positive rate

Using optimal-threshold predictions:

1. Match predictions to GT (same class, IoU >= threshold), greedily by highest IoU.
2. Unmatched predictions are false positives.
3. Normalize by number of predictions:

\[
\text{FP\_rate} = \frac{\#\text{false positives}}{\max(1, \#\text{predictions})}
\]

This keeps FP rate in `[0, 1]`.

### Step 5: Compute image-level missed-detection rate (auxiliary)

Using same matching logic:

\[
\text{Missed\_rate} = \frac{\#\text{unmatched GT}}{\max(1, \#\text{GT})}
\]

This value is saved and used when selecting `w2` in `balance_correlation` mode.

### Step 6: Select `w2` under `balance_correlation`

`w1` is fixed from config (`0.9`).

`w2` is automatically searched over a grid to balance:

- correlation(score, missed_rate)
- correlation(score, fp_rate)

Objective in code:

1. Minimize absolute gap `|corr_miss - corr_fp|`
2. Tie-break by maximizing `(corr_miss + corr_fp)`

Result for this run:

- `w2 = 0.9405517378930963`

(So final `w2` is not the raw config default `0.1` in this mode.)

### Step 7: Final image score `S_img`

For each image:

\[
S_{img} = w_1 \cdot \text{avg\_object\_score} + w_2 \cdot \text{FP\_rate}
\]

with:

- `w1 = 0.9`
- `w2 = 0.9405517378930963`

---

## 3) Real numeric example (first image in score file)

From the first image entry:

- `image_path`: `/home/khanh/Projects/DifficultyAgri/datasets/minneapple/yolo_format/minneapple_yolo/train/images/20150919_174151_image1.png`
- `S_img`: `0.2673193607104153`
- `num_objects`: `95`
- `FP_rate`: `0.04597701149425287`
- `missed_detections_rate`: `0.12631578947368421`

Using

\[
\text{avg\_object\_score} = \frac{S_{img} - w_2\cdot FP\_rate}{w_1}
\]

we get:

- `avg_object_score = 0.24897289182929433`

Recompose check:

\[
0.9\times 0.24897289182929433 + 0.9405517378930963\times 0.04597701149425287 = 0.2673193607104153
\]

which exactly matches saved `S_img`.

---

## 4) Distribution summary from the current score file

Image-level (`S_img`):

- min: `0.09843393494305269`
- max: `0.6009431553478379`
- mean: `0.23287331311095033`
- 75th percentile: `0.26031634941676984`

Object-level (`S_obj`):

- min: `0.04596262738651524`
- max: `1.0`
- mean: `0.22739804182580156`
- 75th percentile: `0.25594720572472895`

---

## 5) Key implementation notes

- `S_obj` is built from **low-conf predictions** to reduce misses and assess object difficulty robustly.
- `FP_rate` is built from **optimal-conf predictions** to represent practical deployment behavior.
- In `balance_correlation`, image score weighting is data-adaptive (`w2` auto-selected).
- Therefore, comparing scores across runs should account for changes in selected `w2`.
