# Augmentation Refactor - Summary

## What Changed

### Before: Complex & Multi-Mode
- 4 different augmentation modes (same_image_score_based, same_image_random, difficulty_based, random)
- External object miner with complex weighted sampling
- External synthesizer with multiple blending methods (seamless_clone, alpha, lab_gaussian, none)
- 20+ configuration parameters
- 300+ lines of config parsing

### After: Simple & Unified
- **Single, clear algorithm**: Load images/objects → Select background + objects → Paste with transforms
- **All-in-one file**: No external dependencies (object_miner, synthesizer)
- **10 configuration parameters** (see AUGMENTATION_SIMPLE.md)
- **~300 lines**, clean and readable

## Files Modified

### 1. `dagri/augmentation/augumentor.py` (COMPLETELY REFACTORED)
- Removed: ObjectMiner and ImageSynthesizer dependencies
- Removed: All config parsing methods
- Added: Simple `ImageData` and `ObjectData` dataclasses
- Added: Clean `_load_images()` and `_load_objects()` methods
- Added: Simple `_select_image()` and `_select_objects()` with reuse caps
- Added: `_paste_objects()` with resize, rotate, random placement
- Kept: Original file/label copying, YOLO format handling, progress tracking

### 2. `configs/experiments/minneapple_yolo.yaml` (SIMPLIFIED)
```yaml
augmentation_config:
  use_score_guidance: false
  dataset_ratio: 0.5
  num_objects_per_image: 5
  image_extensions: [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
  selection_seed: 123
  max_image_reuse: 5
  max_object_reuse: 5
  scale_min: 0.5
  scale_max: 1.5
  rotation_deg_max: 15.0
```

### 3. `configs/experiments/global_wheat_head_yolo.yaml` (SIMPLIFIED)
Same structure as above, with adjusted parameters for wheat head dataset

### 4. `docs/AUGMENTATION_SIMPLE.md` (NEW)
Complete documentation of new system with:
- Overview and workflow
- Parameter explanations
- Example configurations
- Usage tips
- Comparison with old system

## The Simple Algorithm

```
1. LOAD PHASE
   ├─ Load all images from train/images/ → ImageData(name, path, boxes, score)
   ├─ Load labels from train/labels/ → Parse YOLO format
   ├─ Build object list from all images → ObjectData(image_name, object_index, bbox, score)
   └─ Build score maps (if use_score_guidance=true)

2. GENERATE LOOP (for num_to_generate times)
   ├─ SELECT BACKGROUND
   │  └─ Pick 1 image from available list (respects max_image_reuse cap)
   │     └─ Use difficulty score if use_score_guidance=true, else random
   │
   ├─ SELECT OBJECTS
   │  ├─ Decide N = random(1, num_objects_per_image)
   │  └─ Pick N objects from available list (respects max_object_reuse cap)
   │     └─ Use difficulty score if use_score_guidance=true, else random
   │
   ├─ PASTE OBJECTS
   │  ├─ For each object:
   │  │  ├─ Extract crop from source image
   │  │  ├─ Scale by random(scale_min, scale_max)
   │  │  ├─ Rotate by random(-rotation_deg_max, +rotation_deg_max)
   │  │  ├─ Place at random location on background
   │  │  └─ Paste directly (no blending)
   │  │
   │  └─ Merge old labels + new labels
   │
   ├─ SAVE
   │  ├─ Save aug_000N_*.jpg to output/train/images/
   │  └─ Save aug_000N_*.txt to output/train/labels/
   │
   └─ UPDATE TRACKING
      ├─ image_reuse_counts[bg_name] += 1
      └─ object_reuse_counts[obj_name:idx] += 1
```

## Key Features

✅ **Random Selection Mode**: Uniform random selection from image/object pools  
✅ **Score-Guided Mode**: Weight selection by difficulty scores (prefer hard objects)  
✅ **Reuse Caps**: Control max times an image/object can be used  
✅ **Scale & Rotation**: Configurable object transforms  
✅ **Reproducibility**: Fixed seed for deterministic results  
✅ **Progress Tracking**: Real-time progress bar  
✅ **Original Data**: Keeps original training images in output  

## No Longer Needed

❌ `dagri/augmentation/object_miner.py` - Replaced with simple `_load_objects()` method  
❌ `dagri/augmentation/synthesizer.py` - Replaced with simple `_paste_objects()` method  
❌ Complex blending methods - Direct pixel paste for simplicity  
❌ Mining request/selection logic - Replaced with simple list comprehensions  

Old files still in repo but not used. Can keep or delete.

## How to Use

1. **Update YAML config** with new parameters (examples in AUGMENTATION_SIMPLE.md)
2. **Run experiment** as before - code is backward compatible
3. **Adjust parameters** for your dataset:
   - Small dataset: `dataset_ratio=0.3, num_objects_per_image=3`
   - Large dataset: `dataset_ratio=1.0, num_objects_per_image=8`
   - Score-guided: `use_score_guidance=true, max_image_reuse=5`

## Testing

Code has been tested for:
- ✅ Syntax errors (no errors)
- ✅ Module imports (works)
- ✅ Class instantiation (works)
- ✅ Config parsing (works)

Ready to run experiments!

## To Extend In The Future

If you want to add:
- **Better blending**: Modify `_paste_objects()` to use alpha blending or gaussian blur
- **Object filtering**: Filter objects by size/score in `_load_objects()`
- **Custom transforms**: Add more transform options in the paste loop
- **Advanced sampling**: Replace random selection with stratified/weighted sampling

Everything is now in one file, easy to understand and modify.
