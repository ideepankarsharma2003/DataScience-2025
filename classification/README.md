# Image Classification


## MultilabelStratifiedKFold

**MultilabelStratifiedKFold** is a variation of the K-fold cross-validation technique designed for multilabel classification problems. In multilabel classification, each instance can belong to multiple classes, unlike traditional single-label classification where each instance belongs to exactly one class. The key challenge is to ensure that the distribution of labels across the folds is similar, ensuring a fair evaluation of the model’s performance.

### Basic Cross-Validation Recap

In traditional K-fold cross-validation for classification:
1. The dataset is divided into <code>K</code> subsets (or folds).
2. For each fold, the model is trained on <code>K-1</code> folds and tested on the remaining fold.
3. This process is repeated <code>K</code> times, each time with a different fold being used for testing.

The primary objective of cross-validation is to ensure that each data point is used for both training and testing, so the model's performance is evaluated on different subsets of the data.

### Why Stratification Matters in Multilabel Classification

In a **multilabel classification** problem, an instance can belong to multiple labels. If the dataset has a large number of labels or classes, it becomes important to ensure that the distribution of labels across the folds is **balanced**. Otherwise, one fold could end up with a disproportionate number of instances for certain labels, and the evaluation might not reflect the true performance of the model across all labels.

In traditional K-fold, we randomly split the data. However, this may not ensure that the proportion of instances with each label is similar across all folds, leading to skewed or biased evaluation.

### MultilabelStratifiedKFold: Intuition

The **MultilabelStratifiedKFold** method solves this problem by performing the following:

1. **Label-wise Stratification**:  
   For each fold, instead of using random splits, MultilabelStratifiedKFold makes sure that the distribution of each label (across all instances) is as uniform as possible across the folds. This ensures that each fold has a similar percentage of instances for each label.

2. **Stratification Mechanism**:
   - For each label, the instances that are associated with that label are divided into <code>K</code> folds in such a way that each fold gets a proportional representation of instances with that label.
   - This is done by treating each label as a binary classification problem, where an instance is either a member (1) or not (0) of that label.
   - The **stratification** ensures that the proportion of positive instances (1s) and negative instances (0s) for each label is consistent across all folds.



    <h2>Mathematical Breakdown</h2>
    <p>Consider a dataset with <i>N</i> samples and <i>L</i> labels (binary). We have a set of labels for each instance:</p>
    <p>
        <i>y<sub>i</sub> = [y<sub>i1</sub>, y<sub>i2</sub>, ..., y<sub>iL</sub>]</i>
        where <i>y<sub>il</sub> ∈ {0, 1}</i>
    </p>
    <p>Where <i>y<sub>il</sub> = 1</i> means instance <i>i</i> belongs to label <i>l</i>, and <i>y<sub>il</sub> = 0</i> means it does not.</p>

    <h2>Goal:</h2>
    <p>Divide the dataset into <i>K</i> folds such that for each label <i>l</i>, the number of positive instances (i.e., <i>y<sub>il</sub> = 1</i>) is approximately the same in each fold. Similarly, the number of negative instances (i.e., <i>y<sub>il</sub> = 0</i>) is also approximately balanced across folds.</p>

    <h2>Stratification:</h2>
    <ul>
        <li>For each label <i>l</i>, treat it as a binary classification problem where the goal is to have a similar proportion of positive and negative instances in each fold.</li>
        <li>The algorithm finds the instances where <i>y<sub>il</sub> = 1</i> and ensures they are evenly distributed across all folds. The same is done for instances where <i>y<sub>il</sub> = 0</i>.</li>
        <li>This is typically achieved using a technique like group k-fold or a stratification method where the instances are grouped in such a way that each fold gets a fair share of positive and negative instances for each label.</li>
    </ul>

    <h2>Example:</h2>
    <p>Consider a dataset with 6 samples and 3 labels:</p>

    <table border="1">
        <thead>
            <tr>
                <th>Sample</th>
                <th>Label 1</th>
                <th>Label 2</th>
                <th>Label 3</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>1</td>
                <td>1</td>
                <td>0</td>
                <td>1</td>
            </tr>
            <tr>
                <td>2</td>
                <td>0</td>
                <td>1</td>
                <td>0</td>
            </tr>
            <tr>
                <td>3</td>
                <td>1</td>
                <td>1</td>
                <td>0</td>
            </tr>
            <tr>
                <td>4</td>
                <td>1</td>
                <td>0</td>
                <td>0</td>
            </tr>
            <tr>
                <td>5</td>
                <td>0</td>
                <td>1</td>
                <td>1</td>
            </tr>
            <tr>
                <td>6</td>
                <td>1</td>
                <td>1</td>
                <td>1</td>
            </tr>
        </tbody>
    </table>

    <p>Using <code>MultilabelStratifiedKFold</code> with <i>K = 3</i> folds, the stratification would ensure that for each label, the distribution of positive and negative instances is balanced across the 3 folds. This would result in folds like:</p>

    <ul>
        <li>Fold 1: Samples 1, 2, 6 (balanced representation for each label)</li>
        <li>Fold 2: Samples 3, 4, 5</li>
        <li>Fold 3: Remaining samples</li>
    </ul>

    <h2>Mathematical Concept:</h2>
    <p>Mathematically, the idea is to partition the label space such that for each label <i>l</i>, the distribution of instances across folds respects the marginal distribution of the label in the overall dataset:</p>
    <p>
        <i>P(y<sub>il</sub> = 1) in each fold ≈ P(y<sub>il</sub> = 1) in the entire dataset</i>
    </p>
    <p>This means each fold maintains a similar proportion of positive and negative examples for every label.</p>


### Benefits of MultilabelStratifiedKFold

- **Balanced Evaluation**: The evaluation results are more reliable since each fold has a similar distribution of labels, providing a better estimate of model performance.
- **Avoiding Bias**: It reduces the risk of one fold having an overrepresentation of certain labels, which would make the model appear to perform better or worse than it truly does for certain labels.
