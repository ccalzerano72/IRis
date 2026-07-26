# Dataset Preprocessing Scripts

## Overview

This directory contains scripts for preparing image datasets for the restoration task. The pipeline handles image preprocessing (resize/crop), train/validation splitting, and synthetic degradation generation using ARNIQA.

## Scripts

### 1. `preprocess_images.py` - Image Preprocessing

Preprocesses images to square format with consistent dimensions.

**Process**:

1. Resize so that min(width, height) = target_size
2. Center crop to target_size × target_size

**Usage**:

```bash
python script/restoration/dataset_preprocess/preprocess_images.py \
    --source_dir data/raw_images \
    --dest_dir data/processed/clean \
    --target_size 512
```

**Arguments**:

- `--source_dir` (required): Path to source images directory
- `--dest_dir` (required): Path to output directory
- `--target_size` (optional): Target size in pixels (default: 512)

**Features**:

- High-quality interpolation (INTER_AREA for downsampling, INTER_LANCZOS4 for upsampling)
- Maximum quality output (PNG compression 0, JPEG quality 100)
- Maintains directory structure
- Processes all common image formats (.jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp)
- Case-insensitive file extension handling
- Generates `filename_list.txt` with all processed files

---

### 2. `generate_degraded.py` - Degradation Generation

Generates degraded images from clean reference images using ARNIQA degradations.

**Usage**:

```bash
python script/restoration/dataset_preprocess/generate_degraded.py \
    --clean_dir data/processed/clean \
    --degraded_dir data/processed/degraded \
    --distortion_name whitenoise \
    --level 2
```

**Arguments**:

- `--clean_dir` (required): Path to clean reference images
- `--degraded_dir` (required): Path to output degraded images
- `--distortion_name` (required): Name of distortion to apply
- `--level` (required): Distortion level (0-4)

**Available Distortions**:

| Category    | Distortions                                                              |
| ----------- | ------------------------------------------------------------------------ |
| Blur        | gaublur, lensblur, motionblur                                            |
| Color       | colordiff, colorshift, colorsat1, colorsat2                              |
| Compression | jpeg, jpeg2000                                                           |
| Noise       | whitenoise, whitenoiseCC, impulsenoise, multnoise                        |
| Brightness  | brighten, darken                                                         |
| Spatial     | meanshift, jitter, noneccpatch, pixelate                                 |
| Contrast    | quantization, colorblock, highsharpen, lincontrchange, nonlincontrchange |

**Distortion Levels**:

- Level 0: Mild degradation
- Level 1: Light degradation
- Level 2: Medium degradation (recommended for initial training)
- Level 3: Strong degradation
- Level 4: Severe degradation

**Features**:

- Uses ARNIQA ImageDistorter for realistic degradations
- Maintains directory structure
- Generates `filename_list.txt`
- Automatic output clamping to valid range

---

### 3. `prepare_dataset.py` - Complete Pipeline (Recommended)

Main orchestration script that handles the complete dataset preparation workflow.

**Usage**:

```bash
python script/restoration/dataset_preprocess/prepare_dataset.py \
    --source_dir data/raw_images \
    --dest_dir data/processed \
    --distortion_name whitenoise \
    --level 2 \
    --target_size 512
```

**Arguments**:

- `--source_dir` (required): Path to source images directory
- `--dest_dir` (required): Path to destination directory
- `--distortion_name` (optional): Distortion type (default: whitenoise)
- `--level` (optional): Distortion level 0-4 (default: 2)
- `--target_size` (optional): Target image size (default: 512)

**Workflow**:

1. Find all images in source directory
2. Split into train (80%) and validation (20%) sets
3. Preprocess clean images for train set
4. Preprocess clean images for validation set
5. Generate degraded images for train set
6. Generate degraded images for validation set

**Output Structure**:

```
dest_dir/
├── train/
│   ├── clean/          # Preprocessed clean training images
│   └── degraded/       # Degraded training images
└── val/
    ├── clean/          # Preprocessed clean validation images
    └── degraded/       # Degraded validation images
```

**Configuration**:

The script has configurable constants at the top:

```python
TRAIN_SPLIT = 0.8   # 80% train, 20% validation
RANDOM_SEED = 42    # For reproducible splits
TARGET_SIZE = 512   # Default image size
```

**Features**:

- Automatic train/validation split with fixed seed (reproducible)
- Complete end-to-end pipeline
- Progress tracking for each step
- Summary statistics at completion
- Organized output structure ready for training

---

## Quick Start

### Minimal Example (Single Degradation)

For quick testing with a small dataset:

```bash
# Prepare dataset with white noise degradation
python script/restoration/dataset_preprocess/prepare_dataset.py \
    --source_dir data/DIV2K/DIV2K_train_HR \
    --dest_dir data/DIV2K/processed \
    --distortion_name whitenoise \
    --level 2
```

This will create:

- ~800 training images (clean + degraded)
- ~200 validation images (clean + degraded)
- Ready for training in ~5-10 minutes

### Full Pipeline Example

For a complete dataset preparation:

```bash
# Step 1: Prepare main training dataset
python script/restoration/dataset_preprocess/prepare_dataset.py \
    --source_dir data/kadis700k/raw \
    --dest_dir data/kadis700k/processed \
    --distortion_name whitenoise \
    --level 2 \
    --target_size 512

# Step 2: Prepare test sets (optional)
python script/restoration/dataset_preprocess/preprocess_images.py \
    --source_dir data/benchmarks/Set5 \
    --dest_dir data/benchmarks/processed/Set5 \
    --target_size 512

python script/restoration/dataset_preprocess/generate_degraded.py \
    --clean_dir data/benchmarks/processed/Set5 \
    --degraded_dir data/benchmarks/processed/Set5_degraded \
    --distortion_name whitenoise \
    --level 2
```

---

## Advanced Usage

### Multiple Degradation Types

To prepare datasets with different degradations:

```bash
# White noise
python script/restoration/dataset_preprocess/prepare_dataset.py \
    --source_dir data/raw \
    --dest_dir data/processed_whitenoise \
    --distortion_name whitenoise \
    --level 2

# JPEG compression
python script/restoration/dataset_preprocess/prepare_dataset.py \
    --source_dir data/raw \
    --dest_dir data/processed_jpeg \
    --distortion_name jpeg \
    --level 2

# Gaussian blur
python script/restoration/dataset_preprocess/prepare_dataset.py \
    --source_dir data/raw \
    --dest_dir data/processed_blur \
    --distortion_name gaublur \
    --level 2
```

### Custom Train/Val Split

To change the split ratio, edit `prepare_dataset.py`:

```python
# At the top of the file
TRAIN_SPLIT = 0.9   # 90% train, 10% validation
```

### Different Image Sizes

For different target sizes:

```bash
# 768x768 images
python script/restoration/dataset_preprocess/prepare_dataset.py \
    --source_dir data/raw \
    --dest_dir data/processed_768 \
    --target_size 768

# 256x256 images (faster training)
python script/restoration/dataset_preprocess/prepare_dataset.py \
    --source_dir data/raw \
    --dest_dir data/processed_256 \
    --target_size 256
```

---

## Technical Details

### Image Quality

**Interpolation**:

- Downsampling: `cv2.INTER_AREA` (best quality for reduction)
- Upsampling: `cv2.INTER_LANCZOS4` (best quality for enlargement)

**Output Quality**:

- PNG: Compression level 0 (no compression, maximum quality)
- JPEG: Quality 100 (maximum quality)

**Rationale**: Avoid introducing compression artifacts in training data.

### File Handling

**Supported Formats**:

- JPEG: `.jpg`, `.jpeg`
- PNG: `.png`
- BMP: `.bmp`
- TIFF: `.tiff`, `.tif`
- WebP: `.webp`

**Case Sensitivity**:

- Scripts handle both lowercase and uppercase extensions
- Duplicate detection on Windows (case-insensitive filesystem)

### Directory Structure

Scripts maintain the source directory structure in the output:

```
source/
├── category1/
│   ├── image1.jpg
│   └── image2.jpg
└── category2/
    └── image3.jpg

↓ (after preprocessing)

dest/
├── category1/
│   ├── image1.jpg
│   └── image2.jpg
└── category2/
    └── image3.jpg
```

---

## Troubleshooting

### "No images found"

**Problem**: Script reports no images found in source directory.

**Solutions**:

- Check that the path is correct
- Verify images have supported extensions
- Check file permissions

### "Image is too small"

**Problem**: Warning about images being too small (only in old version).

**Solution**: Current version processes all images (including upsampling).

### Duplicate files on Windows

**Problem**: Script processes twice as many images as expected.

**Solution**: Fixed in current version using set-based deduplication.

### Poor image quality

**Problem**: Processed images look blurry or have artifacts.

**Solution**: Current version uses highest quality interpolation and output settings.

### Out of disk space

**Problem**: Not enough space for processed dataset.

**Solutions**:

- Use smaller target_size (e.g., 256 or 384)
- Process a subset of images
- Use external storage

---

## Performance

### Processing Speed

Typical performance on modern hardware (SSD):

| Dataset Size  | Target Size | Time (approx) |
| ------------- | ----------- | ------------- |
| 1,000 images  | 512x512     | 2-5 minutes   |
| 10,000 images | 512x512     | 20-50 minutes |
| 30,000 images | 512x512     | 1-2 hours     |

**Factors affecting speed**:

- Source image size (larger = slower)
- Target size (larger = slower)
- Disk speed (SSD vs HDD)
- CPU performance

### Storage Requirements

Approximate storage for processed datasets:

| Images | Size    | Clean  | Degraded | Total   |
| ------ | ------- | ------ | -------- | ------- |
| 1,000  | 512x512 | ~2 GB  | ~2 GB    | ~4 GB   |
| 10,000 | 512x512 | ~20 GB | ~20 GB   | ~40 GB  |
| 30,000 | 512x512 | ~60 GB | ~60 GB   | ~120 GB |

**Note**: PNG with compression 0 uses more space but preserves maximum quality.

---

## Integration with Training

After preprocessing, the dataset is ready for training:

```python
# In dataset configuration
dataset:
  train:
    clean_dir: data/processed/train/clean
    degraded_dir: data/processed/train/degraded
  val:
    clean_dir: data/processed/val/clean
    degraded_dir: data/processed/val/degraded
```

The `filename_list.txt` files can be used for efficient dataset loading.

---

## Next Steps

After dataset preparation:

1. **Verify data quality**: Visually inspect a few random images
2. **Check statistics**: Verify train/val split sizes
3. **Implement dataset class**: Create PyTorch Dataset for loading
4. **Test data loading**: Verify dataset loader works correctly
5. **Start training**: Begin with small experiments

---

## References

- **ARNIQA**: Degradation generation framework

  - Paper: "ARNIQA: Learning Distortion Manifold for Image Quality Assessment"
  - 25 degradation types with 5 severity levels each

- **OpenCV**: Image processing library
  - Interpolation methods: INTER_AREA, INTER_LANCZOS4
  - High-quality image I/O

---

## Document History

| Date       | Version | Changes               |
| ---------- | ------- | --------------------- |
| 2025-10-19 | 1.0     | Initial documentation |
