# Simplified Copy-Paste Augmentation

## Overview

The refactored augmentation system is simple and straightforward:

1. **Load**: Read all images and their labels → Get lists of images and objects
2. **Select**: Pick one background image (random or score-guided) + pick N objects (random or score-guided)
3. **Paste**: Resize, rotate, and paste objects onto background
4. **Save**: Write augmented image and merged labels

## Configuration

```yaml
augmentation_config:
  # Score-guided (true) vs random selection (false)
  use_score_guidance: false
  
  # Generate this ratio of new images based on training set size
  # E.g., if 100 images and dataset_ratio=0.5, generate 50 new images
  dataset_ratio: 0.5
  
  # Paste between 1 and this many objects per augmented image
  num_objects_per_image: 5
  
  # File extensions to discover
  image_extensions: [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
  
  # RNG seed for reproducibility (null = random each time)
  selection_seed: 123
  
  # Maximum times an image can be used as background (null = unlimited)
  max_image_reuse: 5
  
  # Maximum times an object can be pasted (null = unlimited)
  max_object_reuse: 5
  
  # Scale range for pasting objects (scale the object by this multiplier)
  scale_min: 0.5
  scale_max: 1.5
  
  # Maximum rotation in degrees
  rotation_deg_max: 15.0
```

## Parameters Explained

### use_score_guidance
- `true`: Selection probability weighted by difficulty scores (prefer harder objects/images)
- `false`: Random uniform selection

### dataset_ratio
- Controls how many augmented images to generate
- `num_new_images = int(num_original_images * dataset_ratio)`
- Example: 100 original + 0.5 ratio = 50 new augmented images

### num_objects_per_image
- For each augmented image, randomly paste between 1 and N objects
- Example: `num_objects_per_image=5` → paste 1-5 objects per image

### max_image_reuse / max_object_reuse
- Prevent over-using the same trainining data
- `max_image_reuse=5` means each background image can be used at most 5 times
- `max_object_reuse=5` means each object can be pasted at most 5 times
- `null` or `None` = unlimited reuse

### scale_min / scale_max
- Range for scaling pasted objects
- Example: `scale_min=0.5, scale_max=1.5` → scale by 0.5x to 1.5x

### rotation_deg_max
- Maximum rotation in degrees (applied as ±rotation_deg_max)
- Example: `rotation_deg_max=15` → rotate by -15 to +15 degrees

## Example Configurations

### Conservative Augmentation (Small Dataset)
```yaml
augmentation_config:
  use_score_guidance: false
  dataset_ratio: 0.3
  num_objects_per_image: 3
  max_image_reuse: 4
  max_object_reuse: 4
  scale_min: 0.8
  scale_max: 1.2
  rotation_deg_max: 10.0
```

### Aggressive Augmentation (Large Dataset)
```yaml
augmentation_config:
  use_score_guidance: true
  dataset_ratio: 1.0
  num_objects_per_image: 8
  max_image_reuse: null  # unlimited
  max_object_reuse: null  # unlimited
  scale_min: 0.5
  scale_max: 2.0
  rotation_deg_max: 20.0
```

### Score-Guided (Focus on Hard Cases)
```yaml
augmentation_config:
  use_score_guidance: true
  dataset_ratio: 0.5
  num_objects_per_image: 5
  max_image_reuse: 5
  max_object_reuse: 5
  scale_min: 0.6
  scale_max: 1.4
  rotation_deg_max: 15.0
  selection_seed: 123  # reproducible
```

## Output

The augmentor creates a new dataset directory with:
- `train/images/aug_0001_*.jpg` - augmented images
- `train/labels/aug_0001_*.txt` - merged labels (original + new objects)
- Original training images copied to `train/images/` and `train/labels/`

## Key Differences from Old System

| Aspect | Old | New |
|--------|-----|-----|
| Modes | 4 different modes | 1 simple algorithm |
| Blending | Multiple methods | Direct paste |
| Config | 20+ parameters | 10 parameters |
| Complexity | High | Low |
| Customization | Limited | Easy to extend |

## Typical Workflow

```python
from dagri.augmentation.augumentor import CopyPasteAugmentor

augmentor = CopyPasteAugmentor(config)
new_dataset = augmentor.create_new_dataset(
    initial_dataset_properties=dataset_props,
    scoring_results=scores,
    new_dataset_path="/path/to/augmented"
)
```

## Tips

1. **Start simple**: Use random selection first, then try score-guided
2. **Monitor reuse caps**: If stopped early, increase `max_image_reuse` or `max_object_reuse`
3. **Reproducibility**: Set `selection_seed` for deterministic results
4. **Scale carefully**: Too much scaling can create unrealistic objects
5. **Rotation**: Keep moderate (10-20°) to avoid distortion

## Extending

To add custom blending later:
1. Modify `_paste_objects()` method
2. Or enhance with alpha blending, gaussian blur, etc.

Currently uses direct pixel copy for simplicity and speed.
