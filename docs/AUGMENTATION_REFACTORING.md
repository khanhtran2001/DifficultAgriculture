# Augmentation Module Refactoring Guide

## What Changed?

The augmentation module has been refactored from a single 850+ line `CopyPasteAugmentor` class into a clean, modular architecture.

### Key Improvements

#### 1. **Configuration Validation** (Early and Clear)
```python
# Before: Scattered config parsing throughout create_new_dataset()
use_score = self.config.get("use_score_guidance")
dataset_ratio = float(self.config.get("dataset_ratio"))
# ... 20+ lines of parsing

# After: Centralized, typed config with validation
@dataclass
class AugmentationConfig:
    use_score_guidance: bool = True
    dataset_ratio: float = 0.5
    # ... all params with types and defaults
    
    def validate(self) -> list[str]:
        """Return validation errors, or [] if valid."""
        errors = []
        if self.score_weight_function not in {"linear", "exponential"}:
            errors.append("Invalid score_weight_function")
        return errors
```

**Benefits:**
- Catch config errors early, not during processing
- Clear defaults and types for every parameter
- Single source of truth for config structure
- Easy to add validation rules

#### 2. **Separation of Concerns** (Five Focused Components)

| Component | Responsibility |
|-----------|-----------------|
| `ImageObjectLoader` | Load images, objects, scores from disk |
| `ObjectSelector` | Pick objects/images respecting reuse caps and score guidance |
| `ObjectTransformer` | Scale and rotate object crops |
| `PlacementFinder` | Find valid placement positions |
| `AugmentationPipeline` | Orchestrate the workflow |

**Before:** All logic in one `create_new_dataset()` method (~400 lines)
**After:** Each component ~50-150 lines, single responsibility

**Benefits:**
- Each component is easy to understand
- Easy to test each component independently
- Easy to modify one component without affecting others
- Easy to debug: errors point to specific component

#### 3. **Clear Data Flow**

```
Load Data
  ↓
For each augmented image:
  ├─ Select background image
  ├─ Select objects to paste
  ├─ Transform objects (scale, rotate)
  ├─ Find valid placements
  ├─ Paste onto background
  └─ Save image + labels + metadata
```

**Benefits:**
- Pipeline is understandable at a glance
- Easy to add debug output at each step
- Easy to modify the workflow

#### 4. **Better Documentation**

Each class and method has clear docstrings:
```python
def select_objects(
    self,
    objects: list[ObjectData],
    reuse_counts: dict[str, int],
    max_reuse: int | None,
    num_to_select: int,
    ...
) -> list[ObjectData]:
    """Select multiple objects, respecting reuse caps and avoiding 
    duplicates within one selection."""
```

#### 5. **Type Hints Throughout**

All functions have input and output types, enabling:
- IDE autocomplete
- Early error detection
- Clearer function contracts

---

## How to Use

### 1. Keep Both Versions During Testing

The new version is in `augumentor_v2.py` while the old `augumentor.py` still works.

You can test the new version without breaking anything:

```bash
# Test the new version
python -c "
from dagri.augmentation.augumentor_v2 import CopyPasteAugmentor
from dagri.general.config_manager import ConfigManager

cfg = ConfigManager('configs/experiments/minneapple_yolo.yaml')
aug = CopyPasteAugmentor(cfg.augmentation_config)
# ... test it
"
```

### 2. Replace the Old One When Ready

Once you're confident:
```bash
cp dagri/augmentation/augumentor.py dagri/augmentation/augumentor_backup.py
cp dagri/augmentation/augumentor_v2.py dagri/augmentation/augumentor.py
rm dagri/augmentation/augumentor_v2.py
```

### 3. All Your Controls Are Still There

Your config still works exactly the same:

```yaml
augmentation_config:
  # Selection mode
  use_score_guidance: true
  same_image_only: true
  score_weight_function: linear
  score_alpha: 3.0
  
  # Generation scale
  dataset_ratio: 0.5
  min_objects_per_image: 2
  max_objects_per_image: 8
  
  # ... all your existing parameters
```

---

## Debugging Made Easier

### Before: Hard to debug where things go wrong

```
Error in create_new_dataset() at line 250
→ Was it loading? Selection? Transformation? Placement?
```

### After: Error points to specific component

```
Error in ObjectSelector.select_objects() at line 185
→ Immediately know it's a selection issue
→ Look at ObjectSelector class only (~80 lines)
```

---

## Adding New Features

### Example: Add "minimum score threshold"

**Before:** Edit `create_new_dataset()`, mix with other logic

**After:**
1. Add to `AugmentationConfig`:
   ```python
   @dataclass
   class AugmentationConfig:
       min_score: float = 0.0  # ← add here
   ```

2. Modify `ObjectSelector.select_objects()`:
   ```python
   available = [
       obj for obj in objects
       if (max_reuse is None or reuse_counts.get(...) < max_reuse)
       and obj.score >= self.min_score  # ← add here
   ]
   ```

Done! No need to touch other components.

---

## Component Reference

### AugmentationConfig

**Purpose:** Centralized config with validation  
**How to use:**
```python
config = AugmentationConfig.from_dict(yaml_dict)
errors = config.validate()
if errors:
    print("Config errors:", errors)
```

### ImageObjectLoader

**Purpose:** Load images and objects from disk  
**Main methods:**
- `load_images()` → list of `ImageData`
- `load_objects()` → list of `ObjectData`

### ObjectSelector

**Purpose:** Pick objects/images with score guidance and reuse caps  
**Main methods:**
- `select_image()` → one `ImageData`
- `select_objects()` → list of `ObjectData`

### PlacementFinder

**Purpose:** Find valid positions to paste objects  
**Main method:**
- `find_placement()` → (x, y) or (None, None)

### ObjectTransformer

**Purpose:** Transform object crops  
**Main method:**
- `transform()` → scaled+rotated numpy array

### AugmentationPipeline

**Purpose:** Orchestrate the full workflow  
**Main method:**
- `augment_dataset()` → generates all augmented images

---

## Performance & Correctness

**New version maintains 100% same behavior:**
- Same selection logic
- Same placement logic
- Same reuse cap enforcement
- Same score-guided weighting
- Same metadata output

**Only differences:**
- ✅ Better organized code
- ✅ Easier to read
- ✅ Easier to debug
- ✅ Easier to modify
- ✅ Early config validation
