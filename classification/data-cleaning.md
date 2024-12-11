# Data Cleaning

## `MultilabelStratifiedKFold`


`MultilabelStratifiedKFold` is a cross-validation strategy designed specifically for **multilabel datasets**, where each sample can belong to multiple classes simultaneously. For instance, an image might be tagged as both **"cat"** and **"dog"**. In such cases, maintaining the distribution of multiple labels across folds is crucial for ensuring reliable model evaluation.

---

### Key Features
1. **Maintains Label Distribution:** Preserves the proportion of each label across all classes in each fold, ensuring that the dataset's complexity is retained during cross-validation.
2. **Designed for Multilabel Problems:** Provides a more meaningful evaluation by accounting for multilabel imbalances that traditional stratification methods cannot handle.

---

### Example Dataset

Consider the following dataset with **8 samples** and **3 possible labels** (`A`, `B`, and `C`). Each sample may belong to one or more of these labels:

| **Sample** | **A** | **B** | **C** |
|------------|-------|-------|-------|
| 1          | 1     | 0     | 1     |
| 2          | 0     | 1     | 1     |
| 3          | 1     | 1     | 0     |
| 4          | 1     | 0     | 0     |
| 5          | 0     | 1     | 0     |
| 6          | 1     | 1     | 1     |
| 7          | 0     | 0     | 1     |
| 8          | 1     | 0     | 0     |

Here:
- Sample 1 belongs to labels `A` and `C`.
- Sample 2 belongs to labels `B` and `C`.
- And so on.

---

### Python Code Example

```python
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Sample Data
data = pd.DataFrame({
    'A': [1, 0, 1, 1, 0, 1, 0, 1],
    'B': [0, 1, 1, 0, 1, 1, 0, 0],
    'C': [1, 1, 0, 0, 0, 1, 1, 0]
})

# Convert to numpy array
X = np.arange(len(data)).reshape(-1, 1)  # Features are simply indices of samples
y = data.values  # Multilabel targets

# Initialize MultilabelStratifiedKFold
mskf = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Perform splitting
for fold, (train_idx, test_idx) in enumerate(mskf.split(X, y)):
    print(f"Fold {fold + 1}")
    print("Train indices:", train_idx)
    print("Test indices:", test_idx)
    print("Train labels:\n", data.iloc[train_idx])
    print("Test labels:\n", data.iloc[test_idx])
    print("-" * 40)
```

---

### Output Breakdown

The output of the above code might look as follows:

#### **Fold 1**
- **Train Indices:** `[0, 2, 3, 4, 6]`
- **Test Indices:** `[1, 5, 7]`
- **Train Labels:**  
  | **Sample** | **A** | **B** | **C** |
  |------------|-------|-------|-------|
  | 1          | 1     | 0     | 1     |
  | 3          | 1     | 1     | 0     |
  | 4          | 1     | 0     | 0     |
  | 5          | 0     | 1     | 0     |
  | 7          | 0     | 0     | 1     |

- **Test Labels:**  
  | **Sample** | **A** | **B** | **C** |
  |------------|-------|-------|-------|
  | 2          | 0     | 1     | 1     |
  | 6          | 1     | 1     | 1     |
  | 8          | 1     | 0     | 0     |

---

### How `MultilabelStratifiedKFold` Works

To understand its intuition:
- In a **single-label stratified approach**, stratification ensures that the proportion of a single label (e.g., `A`) remains similar across all folds.
- In a **multilabel approach**, each sample may have multiple labels (`A`, `B`, `C`), so the strategy ensures that the **proportions of all labels** are preserved in each fold.

For example:
- If 50% of the samples in the dataset belong to label `A`, this ratio will be preserved in the training and testing splits of every fold.
- Similarly, the proportions of labels `B` and `C` are preserved.

This method ensures that all folds represent the complexity of the multilabel problem.

---

### Advantages of `MultilabelStratifiedKFold`

1. **Balanced Evaluation:** Ensures each fold has a similar label distribution, preventing any fold from being over- or under-represented with certain labels.
2. **Improved Model Generalization:** By maintaining label diversity in each fold, models trained and validated on these folds are less prone to overfitting specific label distributions.
3. **Essential for Multilabel Tasks:** Traditional stratified methods fail for multilabel problems, as they consider only a single label.

By keeping the multilabel structure intact, `MultilabelStratifiedKFold` is an essential tool for evaluating multilabel models effectively.
