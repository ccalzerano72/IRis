# Restoration Dataset Configuration

## Overview

Hierarchical configuration structure with **3 levels**:

1. **Dataset-specific configs** (`dataset_div2k.yaml`, `dataset_kadis700k.yaml`)

   - Dataset name and paths
   - Split configuration (presplit or auto)

2. **Base config** (`dataset_base.yaml`)

   - Inherits from dataset-specific config
   - Adds degradation settings
   - Adds preprocessing settings

3. **Task configs** (`dataset_train.yaml`, `dataset_val.yaml`, `dataset_vis.yaml`)
   - Inherit from base config
   - Add task-specific paths

## 🎯 To Switch Dataset: Edit ONE Line!

### In `dataset_base.yaml`:

```yaml
# CHANGE THIS LINE to switch dataset:
base_config:
  - config/dataset_restoration/dataset_div2k.yaml # ← Use DIV2K
# - config/dataset_restoration/dataset_kadis700k.yaml  # ← Use Kadis700k
```

**That's it!** Everything else updates automatically.

---

## 📁 File Structure

```
config/dataset_restoration/
├── dataset_div2k.yaml          # DIV2K-specific settings
├── dataset_kadis700k.yaml      # Kadis700k-specific settings
├── dataset_base.yaml           # Inherits from above + adds degradation/resize
├── dataset_train.yaml          # Inherits from base + adds train paths
├── dataset_val.yaml            # Inherits from base + adds val paths
└── dataset_vis.yaml            # Inherits from base + adds vis paths
```

---

## 📋 Configuration Hierarchy

### Level 1: Dataset-Specific

**`dataset_div2k.yaml`:**

```yaml
dataset_name: div2k
dataset_root: div2k

split_config:
  mode: presplit
  train_subfolder: DIV2K_train_HR
  val_subfolder: DIV2K_valid_HR
```

**`dataset_kadis700k.yaml`:**

```yaml
dataset_name: kadis700k
dataset_root: kadis700k

split_config:
  mode: auto
  train_ratio: 0.8
  random_seed: 42
```

### Level 2: Base Configuration

**`dataset_base.yaml`:**

```yaml
base_config:
  - config/dataset_restoration/dataset_div2k.yaml # ← Switch here!

degradation_config:
  mode: pre_generated
  types:
    - whitenoise
  levels: [2]
  mixed_prob: 0.0

resize_to_hw:
  - 512
  - 512
```

### Level 3: Task Configurations

**`dataset_train.yaml`:**

```yaml
base_config:
  - config/dataset_restoration/dataset_base.yaml

dataset:
  train:
    name: restoration_${dataset_name}
    dir: ${dataset_root}/train/clean
    filenames: data_split/restoration/${dataset_name}_train.txt
    # ... inherits resize_to_hw and degradation_config
```

---

## 🔄 Switching Datasets

### Example: Switch from DIV2K to Kadis700k

**Step 1:** Edit `dataset_base.yaml`:

```yaml
base_config:
  # - config/dataset_restoration/dataset_div2k.yaml        # Comment out
  - config/dataset_restoration/dataset_kadis700k.yaml # Uncomment
```

**Step 2:** Run preparation script:

```bash
python script/restoration/dataset_preprocess/prepare_dataset.py \
  --source_dir /path/to/kadis700k/images
```

**Done!** All configs automatically use Kadis700k settings.

---

## ➕ Adding New Dataset

### Example: Add Custom Dataset

**Step 1:** Create `dataset_custom.yaml`:

```yaml
# config/dataset_restoration/dataset_custom.yaml
dataset_name: my_dataset
dataset_root: my_dataset

split_config:
  mode: auto
  train_ratio: 0.9 # 90% train, 10% val
  random_seed: 123
```

**Step 2:** Update `dataset_base.yaml`:

```yaml
base_config:
  - config/dataset_restoration/dataset_custom.yaml
```

**Step 3:** Run script:

```bash
python script/restoration/dataset_preprocess/prepare_dataset.py \
  --source_dir /path/to/my/images
```

---

## 🎯 What Each Config Defines

| Config                   | Defines                        | Example                        |
| ------------------------ | ------------------------------ | ------------------------------ |
| `dataset_div2k.yaml`     | Dataset name, root, split mode | `div2k`, `presplit`            |
| `dataset_kadis700k.yaml` | Dataset name, root, split mode | `kadis700k`, `auto`            |
| `dataset_base.yaml`      | Degradation, preprocessing     | `whitenoise`, `512x512`        |
| `dataset_train.yaml`     | Train paths                    | `div2k/train/clean`            |
| `dataset_val.yaml`       | Val paths                      | `div2k/val/clean`              |
| `dataset_vis.yaml`       | Vis paths                      | `div2k/val/clean` (10 samples) |

---

## 🔧 Customization Examples

### Change Degradation Type

Edit `dataset_base.yaml`:

```yaml
degradation_config:
  types:
    - jpeg # Changed from whitenoise
  levels: [3] # Changed from [2]
```

### Change Image Size

Edit `dataset_base.yaml`:

```yaml
resize_to_hw:
  - 768 # Changed from 512
  - 768
```

### Change Split Ratio (for auto mode)

Edit `dataset_kadis700k.yaml` (or your dataset config):

```yaml
split_config:
  mode: auto
  train_ratio: 0.9 # Changed from 0.8
  random_seed: 42
```

---

## 📊 Variable Interpolation

All configs use `${variable}` syntax:

```yaml
# In dataset_train.yaml:
name: restoration_${dataset_name} # → restoration_div2k
dir: ${dataset_root}/train/clean # → div2k/train/clean
filenames: data_split/restoration/${dataset_name}_train.txt
```

Variables are resolved automatically by OmegaConf after merging all base configs.

---

## ✅ Benefits of This Structure

1. **Single point of change**: Edit one line in `dataset_base.yaml`
2. **Modular**: Each dataset has its own config
3. **Reusable**: Degradation/resize settings shared across datasets
4. **Extensible**: Easy to add new datasets
5. **Consistent**: All task configs automatically use correct dataset

---

## 🚀 Quick Start

### For DIV2K:

1. **Config is already set** (default in `dataset_base.yaml`)
2. **Run script**:
   ```bash
   python script/restoration/dataset_preprocess/prepare_dataset.py \
     --source_dir /path/to/DIV2K
   ```

### For Kadis700k:

1. **Edit `dataset_base.yaml`**: Uncomment kadis700k line
2. **Run script**:
   ```bash
   python script/restoration/dataset_preprocess/prepare_dataset.py \
     --source_dir /path/to/kadis700k/images
   ```

---

## 📝 Notes

- **BASE_DATA_DIR**: Not needed in configs! Script uses env var or `./data` default
- **Paths are relative**: All paths relative to BASE_DATA_DIR
- **File lists**: Generated automatically in `data_split/restoration/`
- **Reproducibility**: Fixed random seed for auto split mode

---

## Document History

| Date       | Version | Changes                                              |
| ---------- | ------- | ---------------------------------------------------- |
| 2025-10-19 | 2.0     | Hierarchical structure with dataset-specific configs |
| 2025-10-19 | 1.0     | Initial documentation                                |
