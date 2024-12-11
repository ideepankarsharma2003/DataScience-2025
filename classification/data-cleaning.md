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

The step `MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)` configures the **MultilabelStratifiedKFold** cross-validation object. Here's a detailed explanation of what happens inside this step:

---

### 1. **`n_splits=3`: Number of Folds**
This specifies that the dataset will be split into **3 folds** for cross-validation. 

- **Cross-validation Concept:** 
  - In each iteration, one fold is used as the **test set**, and the remaining two folds are combined to form the **training set**.
  - With 3 splits, this results in **3 iterations**: each fold is used once as the test set.

---

### 2. **`shuffle=True`: Shuffle the Dataset**
- **Shuffling before Splitting:** 
  - The dataset is shuffled before being divided into folds. This helps to randomize the distribution of samples, which can prevent patterns in the dataset from influencing the split (e.g., consecutive rows having similar labels).
  
- **Effect of Shuffling on Multilabel Stratification:**
  - Even with shuffling, the algorithm ensures that the label proportions are preserved in each fold. Shuffling only affects the order of samples before the stratification process begins.

---

### 3. **`random_state=42`: Reproducibility**
- **Random State as a Seed:** 
  - The `random_state` parameter sets a fixed seed for the random number generator used during shuffling.
  - This ensures that the shuffling process is deterministic and reproducible. 
  - If you run the code multiple times with the same `random_state`, you will get the same splits every time.

---

### Internal Workflow of `MultilabelStratifiedKFold`

When the object is initialized with the above parameters, the following happens internally:

1. **Input Preparation:**
   - The dataset's features (`X`) and multilabel targets (`y`) are analyzed. 
   - The algorithm identifies how many samples belong to each label and calculates the proportions of each label.

2. **Shuffling (if `shuffle=True`):**
   - The dataset is shuffled based on the `random_state` seed. 
   - This ensures that the splitting is random but repeatable.

3. **Stratified Splitting:**
   - The algorithm divides the dataset into `n_splits` (e.g., 3) folds.
   - It ensures that for each fold:
     - The proportion of each label (e.g., `A`, `B`, `C`) in the **test set** is approximately the same as in the original dataset.
     - The remaining data forms the **training set** for that fold while also preserving the label proportions.

4. **Folds Are Ready:**
   - The result is a series of train-test splits, each balanced in terms of the multilabel distribution.

---

### Example: How the Splits Work

Suppose you have 6 samples and 2 labels (A, B). The original distribution of labels is:

| Sample | Label A | Label B |
|--------|---------|---------|
| 1      | 1       | 0       |
| 2      | 0       | 1       |
| 3      | 1       | 1       |
| 4      | 1       | 0       |
| 5      | 0       | 1       |
| 6      | 1       | 1       |

- Label A appears in 4 out of 6 samples (66.7%).
- Label B appears in 4 out of 6 samples (66.7%).

After shuffling and splitting into 3 folds, the proportions of `A` and `B` in each fold remain roughly consistent. For example:

| Fold      | Train Samples        | Test Samples | Label A % in Test | Label B % in Test |
|-----------|----------------------|--------------|--------------------|-------------------|
| Fold 1    | [2, 3, 4, 5]         | [1, 6]       | 50%               | 50%               |
| Fold 2    | [1, 3, 5, 6]         | [2, 4]       | 50%               | 50%               |
| Fold 3    | [1, 2, 4, 6]         | [3, 5]       | 50%               | 50%               |

Notice that the proportions of `A` and `B` in each fold are preserved, even after shuffling.

---

### Why Each Parameter Matters

- **`n_splits`:** Defines the number of train-test splits for thorough model evaluation.
- **`shuffle`:** Prevents any order-based biases in the dataset.
- **`random_state`:** Enables reproducibility, ensuring consistent splits for debugging and comparisons.

By setting `n_splits=3, shuffle=True, random_state=42`, you create a randomized, stratified, and reproducible cross-validation scheme suitable for multilabel datasets.
