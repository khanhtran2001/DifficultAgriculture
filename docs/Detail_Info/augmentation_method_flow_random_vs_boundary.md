# Augmentation Method Flow: Random vs Boundary-Based Selection

## Purpose

This document explains the full flow of the copy-paste augmentation pipeline implemented in:

- dagri/augmentation/augumentor.py
- dagri/augmentation/synthesizer.py
- dagri/augmentation/object_miner.py (related miner utilities)
- experiments/05_copy_paste_exp.py (experiment orchestration)

The goal is to make the reader understand exactly what happens in:

1. Random mode (no score guidance)
2. Boundary mode (score-guided by low/medium/high groups)

---

## 1) Where augmentation sits in the experiment flow

In experiments/05_copy_paste_exp.py, the high-level sequence is:

1. Load and validate dataset.
2. Evaluate baseline model.
3. Generate predictions and compute scoring results.
4. Run copy-paste augmentation using those scores.
5. Train and evaluate model on augmented dataset.

Augmentation starts after scoring:

- scoring = Scorer(scoring_config)
- score_results = scoring.score(...)
- augmentor = CopyPasteAugmentor(augmentation_config)
- new_dataset_properties = augmentor.create_new_dataset(...)

So, boundary mode depends on score_results from Step 3, while random mode can run without score weighting.

---

## 2) Data structures used by the augmentor

From dagri/augmentation/augumentor.py:

- ImageData:
  - name, path
  - boxes (YOLO labels from that image)
  - score (image-level difficulty, optional)

- ObjectData:
  - source image name/path
  - object index in source label file
  - bbox (YOLO format)
  - score (object-level difficulty, optional)

The scoring maps are built from ScoringResults:

- Image score map key: image name or image stem
- Object score map key: image_name:object_index or stem:object_index

---

## 3) Core create_new_dataset flow (common to all modes)

Method: CopyPasteAugmentor.create_new_dataset(...)

### Step A: Prepare output folders

It creates:

- train/images
- train/labels
- train/metadata

inside new_dataset_path.

It removes previous generated files matching aug_* so reruns do not accumulate stale artifacts.

### Step B: Copy original train split

All original train images and labels are copied first.

Important implication:

- Augmented dataset = original train set + generated augmented images.

### Step C: Read config and initialize RNG

Main parameters:

- use_score_guidance
- dataset_ratio
- min_objects_per_image / max_objects_per_image
- same_image_only
- max_image_reuse / max_object_reuse
- scale_min / scale_max
- rotation_deg_max
- min_object_area_px
- blending_method
- avoid_overlap / placement_control / jiggle settings
- selection_seed

Selection logic controls:

- image_selection_method: score or boundary
- image_selection_group: low / medium / high
- object_selection_method: score or boundary
- object_selection_group: low / medium / high

### Step D: Load candidate images and objects

_load_images(...):

- Reads train images by extension.
- Reads matching YOLO labels.
- Adds image score when score guidance is enabled.

_load_objects(...):

- Iterates objects from all image label boxes.
- Computes object area in pixels and optionally filters by min_object_area_px.
- Adds object score when score guidance is enabled.

### Step E: Decide how many augmented images to generate

num_to_generate = max(1, int(len(images) * dataset_ratio))

Example: 1000 training images with dataset_ratio=0.3 -> 300 augmented images.

### Step F: Generate each augmented image

For each i in num_to_generate:

1. Select one background image via _select_image(...)
2. Draw number of pasted objects uniformly between [min_objects_per_image, max_objects_per_image]
3. Build object pool:
   - same_image_only=True -> only objects from that background image
   - False -> all objects from full pool
4. Select object instances via _select_objects(...)
5. Sample transform parameters:
   - scale_factor in [scale_min, scale_max]
   - rotation_deg in [-rotation_deg_max, +rotation_deg_max]
6. Paste objects via _paste_objects(...)
7. Save:
   - augmented image to train/images/aug_XXXX_*.jpg
   - merged labels (original + new boxes) to train/labels
   - provenance metadata JSON to train/metadata
8. Update reuse counters for image and object picks.

If no eligible background images remain (reuse cap reached), generation stops early.

---

## 4) Random mode flow (do not use score)

Random mode means:

- use_score_guidance = false

Then selection functions ignore score values:

### Background selection

_select_image(...):

- Filter by max_image_reuse if set.
- Pick uniformly at random from available images.

### Object selection

_select_objects(...):

- Build available objects under max_object_reuse.
- Pick uniformly at random (with sequential draws while respecting caps).

### What remains non-random

Even in random mode, these controls still apply:

- same_image_only
- min/max objects per image
- object area filter
- overlap avoidance and placement strategy
- scaling and rotation ranges
- blending method
- reuse caps
- RNG seed (selection_seed) for reproducibility

So random mode is random only in selection weighting, not unconstrained free placement.

---

## 5) Boundary mode flow (score-guided by groups)

Boundary mode is activated per selector by setting:

- image_selection_method = boundary
- object_selection_method = boundary

and choosing group:

- low / medium / high

This requires use_score_guidance = true, because boundary grouping uses score values.

### Boundary grouping rule

Method: _filter_by_boundary(...)

For the candidate list (images or objects):

1. Sort by score ascending (low difficulty -> high difficulty).
2. Split list into 3 nearly equal parts.
3. Return one part:
   - low: easiest third
   - medium: middle third
   - high: hardest third

Remainder handling:

- high and medium receive remainder according to implementation slicing.

### Boundary background selection

_select_image(...):

- Apply reuse filter first.
- Filter available images to requested boundary group.
- Choose uniformly random within that group.

### Boundary object selection

_select_objects(...):

- Apply object reuse filter first.
- Filter available objects to requested boundary group.
- Choose uniformly random within that filtered group for each draw.

### Example interpretation

If configured with:

- image_selection_method=boundary, image_selection_group=low
- object_selection_method=boundary, object_selection_group=high

Then each augmented sample tends to:

- use an easier background image
- paste harder objects

This can explicitly shape dataset difficulty curriculum.

---

## 6) Score-weighted mode vs boundary mode (both are score-guided)

When selection_method is score (not boundary):

- weights are computed from normalized score using _scores_to_weights(...)
- supported functions:
  - linear: weight = normalized_score^alpha
  - exponential: weight = exp(alpha * normalized_score) - 1
- reverse_score_guidance=true flips normalized score to (1 - normalized)

Difference:

- score mode: probabilistic preference over all candidates
- boundary mode: hard filtering to one third, then uniform random selection

Boundary is stricter and easier to interpret experimentally.

---

## 7) Paste and geometry flow

Method: _paste_objects(...)

For each selected object:

1. Crop object patch from source image using YOLO bbox.
2. Build object mask and apply transform through synthesizer:
   - resize by scale_factor
   - rotate by rotation_deg
   - crop to transformed mask bounds
3. Find placement:
   - placement_control=True: try jiggle around existing boxes, then random
   - placement_control=False: random first, then jiggle fallback
4. Collision handling:
   - if avoid_overlap=True, reject placements colliding with existing boxes (+margin)
5. Blend and paste:
   - none (hard paste), alpha, lab_gaussian, or seamless_clone
6. Convert pasted pixel bbox back to YOLO normalized coordinates.
7. Append new box and continue.

If no valid placement found for an object, that object is skipped.

---

## 8) Metadata generated per augmented sample

For each augmented image, metadata JSON includes:

- background image identity
- selected object count and pasted object count
- whether score guidance was used
- method settings (reverse_score_guidance, same_image_only, blending_method, score function)
- list of selected source objects and scores
- list of pasted boxes

This makes post-hoc analysis and debugging reproducible.

---

## 9) Practical behavior summary

### Random mode (no score guidance)

- Good as neutral baseline augmentation.
- Preserves all geometric/blending controls.
- Does not push toward easy or hard examples intentionally.

### Boundary mode (score guidance with groups)

- Explicitly controls difficulty region:
  - low: easier subset
  - medium: middle subset
  - high: hardest subset
- Can independently control background difficulty and pasted object difficulty.
- More interpretable than pure weighted sampling when designing experiments.

---

## 10) Minimal config patterns

### Pure random selection

- use_score_guidance: false
- image_selection_method: score (ignored when score guidance is false)
- object_selection_method: score (ignored when score guidance is false)

### Boundary high/high selection

- use_score_guidance: true
- image_selection_method: boundary
- image_selection_group: high
- object_selection_method: boundary
- object_selection_group: high

### Boundary low background + high objects

- use_score_guidance: true
- image_selection_method: boundary
- image_selection_group: low
- object_selection_method: boundary
- object_selection_group: high

This variant is often useful to inject hard objects into easier scenes.

---

## 11) Reproducibility and failure conditions

Reproducibility:

- selection_seed controls deterministic selection and sampling behavior.

Potential early stops or skips:

- no eligible backgrounds left under reuse caps
- no eligible objects after area/reuse/group filters
- invalid transforms or no valid placement due to overlap constraints

The pipeline handles these by skipping objects/images or stopping generation early with logs.
