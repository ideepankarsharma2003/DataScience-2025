# Data Cleaning


## [K-fold-StratifiedSplit](https://arxiv.org/pdf/2308.14466)

### **Think of it like sorting toys in boxes!**

Imagine you have:
1. **Pictures of toys (images)**: These are stored in one folder.
2. **Descriptions of what's in the pictures (text files)**: These are stored in another folder.
3. You want to split all these toys into **k boxes (subsets)** so that each box has a balanced mix of toys, shapes, and sizes.

### **Why is this important?**
We want to teach a toy-sorting robot (a model) how to sort toys. But if one box has only tiny toy cars and another box has only big teddy bears, the robot might get confused. So, we split the toys in a smart way, considering:
1. What types of toys are in each picture (classes).
2. How big the toys are (width and height).
3. The shape of the toys (height-to-width ratio).

---

### **Steps of the Algorithm**
#### 1. **List all images and text files**
We first collect the names of all the image files (toy pictures) and text files (toy descriptions). Sometimes, not all pictures have descriptions.

#### 2. **Create a "data table"**
For each image:
- If it has no description, we mark it as a background image (just like saying, "There's nothing here").
- Otherwise, we read the description and note:
  - What type of toys are there.
  - Their positions, sizes, etc.

#### 3. **Prepare the data for splitting**
- We count how many toys of each type are in every picture.
- We calculate the average size and shape of the toys in each picture.
- This helps us split the pictures fairly.

#### 4. **Split into k groups**
Using a method called "stratified splitting," we divide the pictures into **k groups**. Each group has:
- A balanced mix of toy types, sizes, and shapes.

---

### **Python Code**
Here’s how you could implement this algorithm in Python:

```python
import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def stratify_yolo_dataset(Fimg, Ftxt, k):
    # Step 1: List image files
    Limg = [os.path.splitext(f)[0] for f in os.listdir(Fimg) if f.endswith('.jpg')]
    Limg = list(set(Limg))  # Remove duplicates

    # Step 2: List text files
    Ltxt = [os.path.splitext(f)[0] for f in os.listdir(Ftxt) if f.endswith('.txt')]
    Ltxt = list(set(Ltxt))  # Remove duplicates

    # Step 3: Create data table
    Ldata = []
    for Fname in Limg:
        if Fname not in Ltxt:
            # Background image
            Ldata.append([Fname + '.jpg', -1, None, None, None, None])
        else:
            # Read text file for image
            txt_file_path = os.path.join(Ftxt, Fname + '.txt')
            with open(txt_file_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                # Empty text file
                Ldata.append([Fname + '.jpg', -1, None, None, None, None])
            else:
                for line in lines:
                    parts = line.strip().split()
                    Ldata.append([Fname + '.jpg'] + list(map(float, parts)))

    # Convert to DataFrame
    columns = ['filename', 'class', 'x', 'y', 'w', 'h']
    data = pd.DataFrame(Ldata, columns=columns)

    # Step 4: Preprocess for stratified splitting
    one_hot = pd.get_dummies(data['class'], prefix='class')
    data = pd.concat([data, one_hot], axis=1)
    data['w'] *= 1000  # Scale width and height
    data['h'] *= 1000
    new_df = data.drop(['class', 'x', 'y'], axis=1).groupby('filename').sum().reset_index()
    new_df['count'] = new_df.iloc[:, 1:].sum(axis=1)
    new_df.loc[new_df['count'] == 0, 'count'] = 1  # Avoid division by zero
    new_df['avg_w'] = new_df['w'] / new_df['count']
    new_df['avg_h'] = new_df['h'] / new_df['count']
    new_df['avg_ratio'] = new_df['avg_h'] / new_df['avg_w']
    new_df = new_df.drop(['w', 'h'], axis=1)

    # Step 5: Multi-label stratified splitting
    mskf = MultilabelStratifiedKFold(n_splits=k)
    for train_idx, val_idx in mskf.split(new_df['filename'], new_df.iloc[:, 1:]):
        X_train = new_df.iloc[train_idx]
        X_val = new_df.iloc[val_idx]
        print("Train:", X_train)
        print("Validation:", X_val)

# Example usage
Fimg = "/path/to/image/folder"
Ftxt = "/path/to/text/folder"
k = 3
stratify_yolo_dataset(Fimg, Ftxt, k)
```

---

### **Example**
- You have 6 pictures:
  - `img1.jpg`, `img2.jpg`, `img3.jpg`, `img4.jpg`, `img5.jpg`, `img6.jpg`
- You have 4 descriptions:
  - `img1.txt` says "toy car at (0.1, 0.1), size 0.2x0.3".
  - `img2.txt` says "toy bear at (0.3, 0.4), size 0.4x0.5".
  - `img3.txt` and `img4.txt` are empty.
- The algorithm will:
  - Treat `img3.jpg` and `img4.jpg` as "background".
  - Calculate sizes and shapes of toys in each picture.
  - Split into 3 groups, ensuring balanced toy types, sizes, and shapes.

This ensures your model learns well from diverse and balanced data!

Let’s construct a sample data table to visualize how the data table (`DataFrame`) will look like at various stages.

---

### **Initial Data Table (`data`)**

#### Input Images:
- `img1.jpg`, `img2.jpg`, `img3.jpg`, `img4.jpg`

#### Text Files:
- `img1.txt` contains:
  ```
  0 0.1 0.1 0.2 0.3  # Class 0 (toy car), x=0.1, y=0.1, width=0.2, height=0.3
  1 0.5 0.6 0.3 0.4  # Class 1 (toy bear), x=0.5, y=0.6, width=0.3, height=0.4
  ```
- `img2.txt` contains:
  ```
  1 0.3 0.4 0.4 0.5  # Class 1 (toy bear), x=0.3, y=0.4, width=0.4, height=0.5
  ```
- `img3.txt` is empty (background image).
- `img4.txt` is missing (background image).

The raw data table would look like this:

| filename   | class | x    | y    | w    | h    |
|------------|-------|------|------|------|------|
| img1.jpg   | 0     | 0.1  | 0.1  | 0.2  | 0.3  |
| img1.jpg   | 1     | 0.5  | 0.6  | 0.3  | 0.4  |
| img2.jpg   | 1     | 0.3  | 0.4  | 0.4  | 0.5  |
| img3.jpg   | -1    | None | None | None | None |
| img4.jpg   | -1    | None | None | None | None |

---

### **After Adding One-Hot Encoding**
We perform **one-hot encoding** for the `class` column. This means we create new columns `class_0` and `class_1` to represent the presence of each class.

| filename   | class | x    | y    | w    | h    | class_0 | class_1 |
|------------|-------|------|------|------|------|---------|---------|
| img1.jpg   | 0     | 0.1  | 0.1  | 0.2  | 0.3  | 1       | 0       |
| img1.jpg   | 1     | 0.5  | 0.6  | 0.3  | 0.4  | 0       | 1       |
| img2.jpg   | 1     | 0.3  | 0.4  | 0.4  | 0.5  | 0       | 1       |
| img3.jpg   | -1    | None | None | None | None | 0       | 0       |
| img4.jpg   | -1    | None | None | None | None | 0       | 0       |

---

### **Grouped Data Table (`new_df`)**

We **group by `filename`** to summarize the data for each image. This step aggregates all information for an image into one row.

#### Calculations:
- `count`: Total number of objects in the image (sum of `class_0` and `class_1`).
- `avg_w`: Average width (`w`) per image (scaled).
- `avg_h`: Average height (`h`) per image (scaled).
- `avg_ratio`: Ratio of height to width (`avg_h / avg_w`).

| filename   | class_0 | class_1 | count | avg_w   | avg_h   | avg_ratio |
|------------|---------|---------|-------|---------|---------|-----------|
| img1.jpg   | 1       | 1       | 2     | 0.25    | 0.35    | 1.4       |
| img2.jpg   | 0       | 1       | 1     | 0.4     | 0.5     | 1.25      |
| img3.jpg   | 0       | 0       | 1     | 0.0     | 0.0     | 0.0       |
| img4.jpg   | 0       | 0       | 1     | 0.0     | 0.0     | 0.0       |

---

### **Explanation of Columns**
1. **filename**: Name of the image file.
2. **class_0, class_1**: Count of objects of each class in the image.
3. **count**: Total number of objects in the image. For background images (`img3.jpg`, `img4.jpg`), we set `count` to 1 to avoid division by zero during calculations.
4. **avg_w, avg_h**: Average width and height of objects in the image.
5. **avg_ratio**: Aspect ratio (height-to-width) of objects in the image.

This table is now ready for the **stratified splitting** process.
Let’s explain **stratified splitting** step by step using the example we built. Imagine you’re dividing your toys (data) into **training** and **validation groups** such that each group gets a fair mix of toy types, sizes, and shapes.

---

### **Goal of Stratified Splitting**
We want to divide the dataset (images) into **k groups** (e.g., training and validation). Each group should:
1. Have a balanced number of images.
2. Represent all toy types (classes) equally.
3. Preserve the distribution of toy sizes and shapes (`avg_w`, `avg_h`, `avg_ratio`).

---

### **Steps in Stratified Splitting**
1. **Input Data:**
   - Use the grouped data table (`new_df`) we created earlier:
     ```
     filename   | class_0 | class_1 | count | avg_w   | avg_h   | avg_ratio
     img1.jpg   | 1       | 1       | 2     | 0.25    | 0.35    | 1.4
     img2.jpg   | 0       | 1       | 1     | 0.4     | 0.5     | 1.25
     img3.jpg   | 0       | 0       | 1     | 0.0     | 0.0     | 0.0
     img4.jpg   | 0       | 0       | 1     | 0.0     | 0.0     | 0.0
     ```

2. **Define Labels for Splitting:**
   - We treat the following columns as labels:
     - `class_0`, `class_1`: Ensure all toy types are represented.
     - `avg_w`, `avg_h`, `avg_ratio`: Ensure sizes and shapes are balanced.

3. **Apply Multi-Label Stratified KFold:**
   - Use `MultilabelStratifiedKFold` from the `iterstrat` library. This algorithm ensures that all labels (`class_0`, `class_1`, `avg_w`, etc.) are distributed proportionally in each group.
   - Example: Split the dataset into **2 groups (k=2)**.

---

### **How the Splitting Works**
#### Step 1: Analyze Data Distribution
The algorithm checks how toy types, sizes, and shapes are distributed:
- **Toy types (`class_0`, `class_1`):**
  - `img1.jpg` has 1 toy car (`class_0`) and 1 teddy bear (`class_1`).
  - `img2.jpg` has 0 toy cars (`class_0`) and 1 teddy bear (`class_1`).
  - `img3.jpg` and `img4.jpg` have no toys (background).

- **Sizes and Shapes (`avg_w`, `avg_h`, `avg_ratio`):**
  - `img1.jpg`: Small toys with a tall shape (`avg_ratio = 1.4`).
  - `img2.jpg`: Bigger toy with a less tall shape (`avg_ratio = 1.25`).
  - `img3.jpg` and `img4.jpg`: No toys (`avg_ratio = 0`).

#### Step 2: Split Data into Groups
Using the stratification algorithm, the data is split into two groups:
- **Group 1 (Training Set):**
  - `img1.jpg`: Contains 1 toy car and 1 teddy bear with sizes (avg_w=0.25, avg_h=0.35).
  - `img3.jpg`: Background image (no toys).

- **Group 2 (Validation Set):**
  - `img2.jpg`: Contains 1 teddy bear with sizes (avg_w=0.4, avg_h=0.5).
  - `img4.jpg`: Background image (no toys).

---

### **Final Split Results**
Here’s how the groups look:

#### Training Set:
| filename   | class_0 | class_1 | count | avg_w   | avg_h   | avg_ratio |
|------------|---------|---------|-------|---------|---------|-----------|
| img1.jpg   | 1       | 1       | 2     | 0.25    | 0.35    | 1.4       |
| img3.jpg   | 0       | 0       | 1     | 0.0     | 0.0     | 0.0       |

#### Validation Set:
| filename   | class_0 | class_1 | count | avg_w   | avg_h   | avg_ratio |
|------------|---------|---------|-------|---------|---------|-----------|
| img2.jpg   | 0       | 1       | 1     | 0.4     | 0.5     | 1.25      |
| img4.jpg   | 0       | 0       | 1     | 0.0     | 0.0     | 0.0       |

---

### **Why This Split is Balanced**
1. **Toy Types (`class_0`, `class_1`)**:
   - Training Set: 1 toy car, 1 teddy bear.
   - Validation Set: 0 toy cars, 1 teddy bear.
   - Both groups represent the distribution of toy types well.

2. **Sizes and Shapes (`avg_w`, `avg_h`, `avg_ratio`)**:
   - Training Set: Small-to-medium toys with one tall shape (`avg_ratio = 1.4`).
   - Validation Set: Medium-sized toy with a slightly less tall shape (`avg_ratio = 1.25`).
   - This ensures the size/shape diversity is preserved.

3. **Background Images**:
   - Both groups get 1 background image each, ensuring balance in non-toy data.

---

### **Summary**
Stratified splitting ensures that each group (training and validation):
- Contains a proportional mix of toy types.
- Has a fair distribution of sizes and shapes.
- Balances special cases like background images.

This process makes sure the model learns effectively from balanced data!
