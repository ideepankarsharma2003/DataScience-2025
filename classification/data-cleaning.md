# Data Cleaning

## `MultilabelStratifiedKFold`

`MultilabelStratifiedKFold` is a cross-validation strategy designed for datasets with multilabel data. In a multilabel setting, each instance can belong to multiple classes simultaneously. For example, an image might be labeled as both "cat" and "dog." Ensuring that the distribution of multiple labels is preserved across folds is crucial in such scenarios.

Unlike traditional `StratifiedKFold`, which only preserves the distribution of a single label, `MultilabelStratifiedKFold` preserves the proportion of **multiple labels** in each fold.

### Key Features
1. **Maintains Label Distribution:** Ensures that the proportion of each label across all classes is approximately the same in each fold.
2. **Useful for Multilabel Problems:** Helps maintain the complexity and balance of multilabel datasets during cross-validation.

---

### Example Dataset

Imagine we have the following dataset with 8 samples and 3 possible labels (columns A, B, C):

| Sample | A   | B   | C   |
|--------|-----|-----|-----|
| 1      | 1   | 0   | 1   |
| 2      | 0   | 1   | 1   |
| 3      | 1   | 1   | 0   |
| 4      | 1   | 0   | 0   |
| 5      | 0   | 1   | 0   |
| 6      | 1   | 1   | 1   |
| 7      | 0   | 0   | 1   |
| 8      | 1   | 0   | 0   |

Here, each sample can belong to one or more labels (A, B, C).

---

### Python Code Example

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

For the above dataset, the splits might look like this:

**Fold 1:**
- Train indices: `[0, 2, 3, 4, 6]`
- Test indices: `[1, 5, 7]`

**Fold 2:**
- Train indices: `[1, 3, 4, 5, 7]`
- Test indices: `[0, 2, 6]`

**Fold 3:**
- Train indices: `[0, 1, 2, 5, 7]`
- Test indices: `[3, 4, 6]`

Each fold preserves the proportion of labels across the training and test sets. For instance, if label `A` is present in 50% of the dataset, the folds maintain approximately the same ratio.

---

### Advantages

- Prevents label imbalance in folds.
- Suitable for multilabel datasets where traditional stratification fails.

By ensuring balanced label distribution, `MultilabelStratifiedKFold` helps models generalize better when dealing with multilabel datasets.
