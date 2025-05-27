<div>
<img src="https://github.com/user-attachments/assets/ccfe7b7c-440c-4508-af10-4ac9fc0ee443" width=1000>
<div>


<div align="center" style="font-family: 'Times New Roman', Times, serif; color: #3498db;">
  <h1 style="font-size: 10em; margin-bottom: 180px;">
    Project Machine Learning :News Headline Category Classification 
    using 'Logistic Regression' and 'DistilBERT Fine-Tuning'
  </h1>
</div>

<div align="right" style="font-family: 'Times New Roman', Times, serif; 
                          color: #8e44ad;
                          font-style: italic;
                          font-weight: bold;
                          margin-right: 15%;
                          margin-top: 10px;
                          font-size: 20em;">
   Realised by : SABI Houssame & CHADADI Zakaria
</div>

## DEMO

<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face Logo" width="30" style="vertical-align: middle;"/> : [TEST DEMO](https://huggingface.co/spaces/SABI202/Category_Headline_classifier)



<!-- Le reste de votre contenu ici -->




---


## 📚 Table of Contents

- [⚙️ Code Description](#code-description)
  - [Loading-a-json-dataset-using-pandas](#loading-a-json-dataset-using-pandas)
  - [Initial-data-loading-and-exploration-with-pandas](#initial-data-loading-and-exploration-with-pandas)
  - [Selecting-relevant-columns-for-analysis](#selecting-relevant-columns-for-analysis)
  - [Visualizing-the-distribution-of-news-categories](#visualizing-the-distribution-of-news-categories)
  - [Grouping-news-categories-into-broader-themes](#grouping-news-categories-into-broader-themes)
  - [Checking for Missing or Empty Values in the Dataset](#checking-for-missing-or-empty-values-in-the-dataset)
  - [Visualizing the Distribution of News Categories](#visualizing-the-distribution-of-grouped-news-categories)
  - [Saving the Trained Classification Pipeline with Pickle](#saving-the-trained-classification-pipeline-with-pickle)
  - [Importing Libraries for a Text Classification Pipeline Using TF-IDF and Logistic Regression](#importing-libraries-for-a-text-classification-pipeline-using-tf-idf-and-logistic-regression)
  - [Preparing Text and Labels for TF-IDF-Based Classification](#preparing-text-and-labels-for-tf-idf-based-classification)
  - [Building a TF-IDF + Logistic Regression Pipeline for Multiclass Text Classification](#building-a-tf-idf--logistic-regression-pipeline-for-multiclass-text-classification)
  - [Training the TF-IDF + Logistic Regression Model](#training-the-tf-idf--logistic-regression-model)
  - [Evaluating Model Performance on the Test Set](#evaluating-model-performance-on-the-test-set)
  - [Computing and Displaying the Confusion Matrix](#computing-and-displaying-the-confusion-matrix)
  - [Calculating Overall TP, FP, FN, and TN from the Confusion Matrix](#calculating-overall-tp-fp-fn-and-tn-from-the-confusion-matrix)
  - [Saving the Trained Logistic Regression Pipeline with Pickle](#saving-the-trained-logistic-regression-pipeline-with-pickle)
  - [Fine-Tuning DistilBERT for Multiclass Text Classification with Hugging Face Transformers](#fine-tuning-distilbert-for-multiclass-text-classification-with-hugging-face-transformers)
  - [Saving the Trained Label Encoder with Joblib](#saving-the-trained-label-encoder-with-joblib)
  - [Loading the Fine-Tuned BERT Model and Performing Inference on New Text](#loading-the-fine-tuned-bert-model-and-performing-inference-on-new-text)
  - [Zipping the Trained Model Directory for Export or Deployment](#zipping-the-trained-model-directory-for-export-or-deployment)
  - [Checking the Details of the Compressed Model Archive](#checking-the-details-of-the-compressed-model-archive)



---





# Code Description
## Loading a JSON Dataset Using Pandas
<div>
 <img src="https://github.com/user-attachments/assets/559d0c2f-e6af-43aa-96aa-b75c5297c65a" width=1000>
 </div>
 <br>
This repository contains a Python script for loading and analyzing a news category dataset in JSON format.

## Initial Data Loading and Exploration with Pandas
<div>
 <img src="https://github.com/user-attachments/assets/1eaf3e59-ad65-4575-8d12-5e1e24e5f907" width=1000>
 </div>
 <br>

The script performs initial data exploration on a news category dataset using Pandas:

1. `data.head()` - Displays the first 5 rows of the DataFrame, providing a quick preview of the dataset structure and sample values.

2. `data.info()` - Shows the DataFrame's:
   - Column names and data types
   - Number of non-null values
   - Memory usage
   This helps understand the dataset's structure and identify potential missing values.

3. `data.describe()` - Generates descriptive statistics for numerical columns including:
   - Count
   - Mean
   - Standard deviation
   - Minimum/maximum values
   - Quartile ranges
   This provides a statistical overview of the numerical features.

These methods form the essential first steps in any data analysis workflow, helping analysts understand their dataset before deeper exploration.

## Selecting Relevant Columns for Analysis

<div>
 <img src="https://github.com/user-attachments/assets/d253ae40-873c-49d7-a157-862e23d13050" width=1000>
 </div>
 <br>

This operation:
- Creates a new DataFrame `df` containing only three selected columns from the original dataset:
  1. `headline` - The title of the news article
  2. `category` - The classification category of the article
  3. `short_description` - A brief summary of the article's content


## Visualizing the Distribution of News Categories
<div>
 <img src="https://github.com/user-attachments/assets/81998a23-c967-4141-bf88-192194afdc33" width=1000>
 </div>
 <br>

This visualization:
- Creates a **countplot** showing the distribution of news articles across categories
- Uses Seaborn for the visualization and Matplotlib for customization
- Orders categories by frequency (highest count first)
- Includes proper labels and title for clarity
- Rotates x-axis labels 90° for better readability
- Adjusts layout to prevent label clipping


## Grouping News Categories into Broader Themes
<div>
 <img src="https://github.com/user-attachments/assets/5d17d3fc-bacf-48ff-a4c3-7b10485686fc" width=1000>
 </div>
 <br>


**This transformation:**
- Consolidates 50+ original categories into 12 broader semantic groups
- Standardizes similar categories with different naming conventions
- Maintains original category names when no mapping exists (via `.fillna()`)

**Benefits**
1. **Simplified Analysis**: Reduces category cardinality for clearer visualizations
2. **Better Generalization**: Groups related topics for more robust modeling
3. **Improved Readability**: Creates more meaningful category labels
4. **Consistent Taxonomy**: Resolves naming inconsistencies in the raw data

**Example Transformations:**
- All news variants → "NEWS"
- Arts, style, culture → "ARTS & CULTURE"
- Health, food, travel → "LIFESTYLE"
- Politics and business → "POLITICS & BUSINESS"

**Usage Notes:**
- The `.fillna()` preserves unmapped categories
- Original 'category' column remains unchanged
- New 'category_grouped' column contains consolidated categories


## Checking for Missing or Empty Values in the Dataset
<div>
 <img src="https://github.com/user-attachments/assets/f9b18b98-6c5e-4cb2-80dc-e97c822fec4d" width=1000>
 </div>
 <br>

This code performs a comprehensive check for missing or empty values in the DataFrame by:
1. Checking for standard null values with `isnull()`
2. Checking for empty strings with `(df == "")`
3. Combining both checks with a logical OR (`|`)
4. Counting occurrences per column with `sum()`

**Output Interpretation**
The output shows:
- Count of null/empty values for each column
- Helps identify columns needing data cleaning
- Reveals potential data quality issues

**Typical Next Steps:**
1. **Drop missing values**: `df.dropna()`
2. **Fill missing values**: `df.fillna(value)`
3. **Investigate patterns**: Check if missingness correlates with other variables


## Visualizing the Distribution of Grouped News Categories
<div>
 <img src="https://github.com/user-attachments/assets/87759644-c085-47a2-ac07-b91bb3eab050" width=1000>
 </div>
 <br>

1. **Ordered Display**  
   - Categories sorted by frequency (highest to lowest)  
   - `value_counts().index` ensures proper sorting

2. **Optimal Layout**  
   - 90° rotated x-labels for readability  
   - `tight_layout()` prevents label clipping

3. **Statistical Insight**  
   - Reveals distribution after category consolidation  
   - Highlights dominant and rare topic groups

**Analytical Value:**
- Identifies imbalanced classes that may affect modeling
- Verifies effectiveness of category grouping strategy
- Provides baseline for content distribution understanding

**Expected Output:**
A bar chart showing:
- NEWS as typically the most frequent category
- LIFESTYLE and ENTERTAINMENT as other major groups
- Long-tail distribution of specialized topics

**Recommended Actions:**
1. For classification: Consider oversampling rare categories
2. For EDA: Compare with original category distribution
3. For reporting: Highlight dominant content areas


## Saving the Trained Classification Pipeline with Pickle
<div>
 <img src="https://github.com/user-attachments/assets/d42d3627-b38c-4933-93d3-626eb597428a" width=1000>
 </div>
 <br>

Serializes the trained classification pipeline to disk for:
- Future predictions without retraining
- Model deployment to production
- Sharing/reproducing results

**Technical Details**
1. **File Format**: `Binary pickle file (.pkl)`
2. **Contents**:
   - Complete trained pipeline (embedding → SMOTE → XGBoost)
   - Custom `TransformerEmbedding` class
   - Learned parameters and weights
3. **Serialization**: Python's native `pickle` protocol



## Importing Libraries for a Text Classification Pipeline Using TF-IDF and Logistic Regression
<div>
 <img src="https://github.com/user-attachments/assets/58ee863c-425b-4408-860f-3cae00c8f09e" width=1000>
 </div>
 <br>

**1. Data Preparation**
- **Label Encoding**: 
  ```python
  le = LabelEncoder()
  y = le.fit_transform(df['category'])
  ```
  Converts text labels to numerical values

**2. Text Vectorization**
- **TF-IDF Vectorizer**:
  - Converts raw text to numerical features
  - Weights words by importance (frequency in document vs corpus)
  - Configurable parameters (ngram_range, max_features, etc.)

**3. Model Training**
- **Logistic Regression**:
  - Linear classifier with probability outputs
  - Suitable for multi-class problems (multinomial option)
  - L1/L2 regularization support

**4. Evaluation**
- **Classification Report**:
  ```python
  print(classification_report(y_test, y_pred))
  ```
  Shows precision, recall, f1-score per class


## Preparing Text and Labels for TF-IDF-Based Classification
<div>
 <img src="https://github.com/user-attachments/assets/655ab4f6-a071-4876-9956-a946580dec18" width=1000>
 </div>
 <br>


- **Combines** headline and short description into single text features
- **Handles missing values** by filling NA/empty strings with empty string
- **Result**: Each sample contains full contextual information in one string

**Label Encoding**
```python
le = LabelEncoder()
y = le.fit_transform(df['category_grouped'])
```
- **Converts text labels** (e.g., "POLITICS", "SPORTS") to numerical values
- **Preserves mapping** in `le.classes_` for inverse transformation later
- **Example encoding**:
  ```python
  print(le.classes_)  # Shows original category names
  print(le.transform(["NEWS", "SPORTS"]))  # Shows encoded values
  ```

**Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```
- **20% test set** (configurable via `test_size`)
- **Stratified sampling** maintains original class distribution
- **Random state fixed** for reproducibility
- **Output shapes**:
  - X_train: (n_samples * 0.8,)
  - X_test: (n_samples * 0.2,) 


## Building a TF-IDF + Logistic Regression Pipeline for Multiclass Text Classification
<div>
 <img src="https://github.com/user-attachments/assets/80a99c70-2e10-4b75-8438-c99afb5d9235" width=1000>
 </div>
 <br>

**Pipeline Components**

1. **TF-IDF Vectorizer** (`tfidf`):
   - `max_features=10000`: Limits vocabulary to top 10,000 terms by frequency
   - `ngram_range=(1, 2)`: Captures both single words and word pairs
   - *Automatically handles*:
     - Tokenization
     - Stop word removal
     - TF-IDF weighting

2. **Multinomial Logistic Regression** (`clf`):
   - `multi_class='multinomial'`: Uses Softmax regression for multi-class
   - `solver='lbfgs'`: Recommended for medium-sized datasets
   - `max_iter=1000`: Ensures convergence
   - `random_state=42`: Reproducible results

**Technical Specifications**

| Component | Key Features | Recommended For |
|-----------|-------------|-----------------|
| TF-IDF | 10K features, uni+bigrams | Medium-sized text corpora |
| Logistic Regression | L2 regularization (default) | 10-1000 classes |

**Performance Optimization Tips**

1. **For Larger Datasets**:
   ```python
   TfidfVectorizer(max_features=50000, min_df=5)
   ```

2. **For Better Accuracy**:
   ```python
   LogisticRegression(class_weight='balanced')
   ```

3. **For Faster Training**:
   ```python
   LogisticRegression(solver='sag', max_iter=500)
   ```

## Training the TF-IDF + Logistic Regression Model
<div>
 <img 
src="https://github.com/user-attachments/assets/0c90ceb1-7e6e-4c4c-b5ee-6c9ef89692be" width=1000>
 </div>
 <br>

**Training Process**

1. **Automatic Feature Transformation**:
   - Text data (`X_train`) is automatically converted to TF-IDF features
   - Creates a vocabulary of 10,000 uni+bigrams (as configured)
   - Applies IDF weighting across all documents

2. **Logistic Regression Optimization**:
   - Solves multinomial classification via L-BFGS optimizer
   - Uses cross-entropy loss with L2 regularization (default)
   - Runs for maximum 1000 iterations or until convergence

**Expected Output**
- No explicit output (returns fitted pipeline object)
- Progress can be monitored with `verbose` parameter:
  ```python
  LogisticRegression(..., verbose=1)
  ```

## Evaluating Model Performance on the Test Set
<div>
 <img src="https://github.com/user-attachments/assets/5d5bcdc7-36f7-42d2-8e5e-fb3e02b7c97d" width=1000>
 </div>
 <br>

**Evaluation Process**

1. **Prediction Generation**:
   - Automatically applies the same TF-IDF transformation to test data
   - Uses the trained logistic regression coefficients
   - Outputs predicted class labels (encoded values)

2. **Classification Report**:
   - Shows key metrics per class using original label names
   - Includes weighted averages across all classes

**Sample Output Structure**

```
                      precision    recall  f1-score   support

            BUSINESS       0.85      0.82      0.83       320
             POLITICS       0.88      0.90      0.89       450
              SPORTS       0.93      0.95      0.94       280
...
        accuracy                           0.88      2000
       macro avg       0.87      0.86      0.87      2000
    weighted avg       0.88      0.88      0.88      2000
```

**Key Metrics Explained**

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| Precision | TP / (TP + FP) | Accuracy of positive predictions |
| Recall | TP / (TP + FN) | Coverage of actual positives |
| F1-score | 2*(Precision*Recall)/(Precision+Recall) | Harmonic mean of precision/recall |
| Support | - | Number of actual occurrences |

**Advanced Evaluation Options**

1. **Confusion Matrix**:
   ```python
   from sklearn.metrics import ConfusionMatrixDisplay
   ConfusionMatrixDisplay.from_predictions(
       y_test, 
       y_pred,
       display_labels=le.classes_,
       xticks_rotation=45
   )
   ```

2. **Class-Specific Thresholds**:
   ```python
   y_proba = pipeline.predict_proba(X_test)
   adjusted_preds = (y_proba[:, 1] > 0.7).astype(int)  # Example for binary
   ```

3. **Custom Metrics**:
   ```python
   from sklearn.metrics import make_scorer
   from sklearn.model_selection import cross_val_score
   ```


## Computing and Displaying the Confusion Matrix
<div>
 <img src="https://github.com/user-attachments/assets/c15a1d6e-9bcd-41b9-a7c5-115ecaaae2b7" width=1000>
 </div>
 <br>

**Interpretation Guide**

1. **Matrix Structure**:
   - Rows represent actual classes
   - Columns represent predicted classes
   - Diagonal shows correct predictions

2. **Sample Output**:
   ```
   [[120   5   3]
    [  2  95  10] 
    [  1   8 130]]
   ```

**Enhanced Visualization (Recommended)**

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(cm, 
            annot=True, 
            fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

**Key Metrics Derived from Confusion Matrix**

1. **Class-Specific Accuracy**:
   ```python
   class_accuracy = cm.diagonal()/cm.sum(axis=1)
   ```

2. **Misclassification Patterns**:
   ```python
   off_diag = cm.sum() - cm.trace()  # Total errors
   ```

3. **Common Error Types**:
   - False Positives (FP): Column sums - diagonal
   - False Negatives (FN): Row sums - diagonal



## Calculating Overall TP, FP, FN, and TN from the Confusion Matrix
<div>
 <img src="https://github.com/user-attachments/assets/0178f1ad-1a0b-40b9-adb9-d2ea05350e3d" width=1000>
 </div>
 <br>

**Interpretation Guide**

1. **Per-Class Metrics** (Multiclass Context):
   - `TP`: Correctly predicted instances per class
   - `FP`: Instances wrongly predicted as the class
   - `FN`: Instances of the class predicted as others
   - `TN`: Instances correctly rejected from the class

2. **Aggregate Metrics**:
   - `total_TP`: All correct positive predictions
   - `total_FP`: All false alarms across classes
   - `total_FN`: All missed positives
   - `total_TN`: All correct rejections

**Derived Metrics Table**

| Metric | Formula | Multiclass Interpretation |
|--------|---------|---------------------------|
| Accuracy | (TP+TN)/Total | Overall correct predictions |
| Precision | TP/(TP+FP) | Class prediction purity |
| Recall | TP/(TP+FN) | Class coverage |
| Specificity | TN/(TN+FP) | Negative prediction accuracy |

**Implementation Notes**

1. **For Imbalanced Classes**:
   ```python
   print("\n=== Per-Class Breakdown ===")
   for i, class_name in enumerate(le.classes_):
       print(f"\nClass {class_name}:")
       print(f"  Precision: {TP[i]/(TP[i]+FP[i]):.2f}")
       print(f"  Recall:    {TP[i]/(TP[i]+FN[i]):.2f}")
   ```

2. **Error Rate Calculation**:
   ```python
   error_rate = (total_FP + total_FN) / cm.sum()
   print(f"\nError Rate: {error_rate:.2%}")
   ```

**Visualization Enhancement**

```python
error_matrix = np.zeros_like(cm)
np.fill_diagonal(error_matrix, TP)
error_matrix = cm - error_matrix

plt.figure(figsize=(12,8))
sns.heatmap(error_matrix,
            annot=True,
            fmt='d',
            cmap='Reds',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Error Matrix (Non-Diagonal Elements)')
plt.show()
```

**Actionable Insights**

1. **High FP Cases**:
   - Review feature importance for mispredicted classes
   - Consider increasing classification thresholds

2. **High FN Cases**:
   - Check for under-represented features
   - Evaluate need for data augmentation



## Saving the Trained Logistic Regression Pipeline with Pickle
<div>
 <img src="https://github.com/user-attachments/assets/28571c07-1cba-4603-8b11-412e1eeb0ed2" width=1000>
 </div>
 <br>

**What This Does**

1. **Saves Entire Pipeline**:
   - TF-IDF vectorizer configuration
   - Trained logistic regression model
   - All preprocessing steps
   - Class label mappings (through the pipeline)

2. **File Format**: Binary pickle (.pkl)


## Fine-Tuning DistilBERT for Multiclass Text Classification with Hugging Face Transformers

```from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import torch

# ===== 1. Preprocessing =====
df['text'] = df['headline'].fillna('') + ' ' + df['short_description'].fillna('')
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category_grouped'])
num_classes = len(label_encoder.classes_)

# ===== 2. Dataset split =====
train_df, test_df = train_test_split(df[['text', 'label']], test_size=0.2, stratify=df['label'], random_state=42)
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
    'test': Dataset.from_pandas(test_df.reset_index(drop=True))
})

# ===== 3. Tokenization =====
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize_fn(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)

dataset = dataset.map(tokenize_fn, batched=True)
dataset = dataset.remove_columns(['text'])
dataset.set_format('torch')

# ===== 4. Load DistilBERT model =====
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)

# ===== 5. Training arguments =====
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    num_train_epochs=3,  # Start small
    per_device_train_batch_size=16,  # Increase for speed
    per_device_eval_batch_size=16,
    weight_decay=0.1,
    logging_dir='./logs',
    logging_steps=50,
    label_smoothing_factor=0.1,
    fp16=True  # Mixed precision training on GPU
)

# ===== 6. Metrics function =====
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ===== 7. Trainer setup =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics
)

# ===== 8. Train the model =====
trainer.train()

# ===== 9. Save final model =====
trainer.save_model("./results/best_model")
tokenizer.save_pretrained("./results/best_model")
```

**1. Data Preparation**
```python
# Combine text features
df['text'] = df['headline'].fillna('') + ' ' + df['short_description'].fillna('')

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category_grouped'])
num_classes = len(label_encoder.classes_)
```

**2. Dataset Splitting**
```python
train_df, test_df = train_test_split(
    df[['text', 'label']], 
    test_size=0.2, 
    stratify=df['label'],
    random_state=42
)

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'test': Dataset.from_pandas(test_df)
})
```

**3. Tokenization**
```python
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_fn(example):
    return tokenizer(
        example['text'],
        padding='max_length',
        truncation=True,
        max_length=128  # Optimal for news headlines+descriptions
    )

dataset = dataset.map(tokenize_fn, batched=True)
```

**4. Model Initialization**
```python
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_classes
)
```

**5. Training Configuration**
```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    fp16=True,  # GPU acceleration
    evaluation_strategy='epoch',
    save_strategy='epoch'
)
```

**6. Evaluation Metrics**
```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    return {'accuracy': accuracy_score(labels, preds),
            'f1': f1,
            'precision': precision,
            'recall': recall}
```

**7. Training Execution**
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics
)

trainer.train()
```

**Performance Optimization Tips**

1. **Memory Efficiency**:
   ```python
   TrainingArguments(..., gradient_accumulation_steps=2)
   ```

2. **Precision**:
   ```python
   TrainingArguments(..., fp16_full_eval=True)
   ```

3. **Early Stopping**:
   ```python
   from transformers import EarlyStoppingCallback
   trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
   ```

**Deployment Ready Outputs**

```python
# Save model artifacts
trainer.save_model("./results/best_model")
tokenizer.save_pretrained("./results/best_model")

# For production loading:
from transformers import pipeline
classifier = pipeline(
    "text-classification",
    model="./results/best_model",
    tokenizer="./results/best_model"
)
```

**Expected Performance**

| Metric | Baseline (TF-IDF) | DistilBERT |
|--------|------------------|------------|
| Accuracy | 0.82 | 0.89 |
| F1-score | 0.81 | 0.88 |
| Inference Speed | 1000 docs/sec | 200 docs/sec |



## Saving the Trained Label Encoder with Joblib
<div>
 <img src="https://github.com/user-attachments/assets/fcc342d6-9860-46f4-bea0-9409ba2d95d9" width=1000>
 </div>
 <br>

**Purpose**
Persists the label encoding scheme to:
- Decode model predictions back to original categories
- Ensure consistent label mapping during deployment
- Maintain reproducibility across environments

**Technical Details**
1. **File Format**: Joblib binary (.pkl)
   - More efficient than pickle for scikit-learn objects
   - Better handling of numpy arrays

2. **Saved Artifacts**:
   - Class-to-index mappings (`label_encoder.classes_`)
   - All fitted transformer attributes

**Loading for Inference**
```python
loaded_encoder = joblib.load('./results/best_model/label_encoder.pkl')

# Example usage
predictions = model.predict(["New AI breakthrough"])
decoded_predictions = loaded_encoder.inverse_transform(predictions)
```


## Loading the Fine-Tuned BERT Model and Performing Inference on New Text
<div>
 <img src="https://github.com/user-attachments/assets/8bdf47cc-49bc-4632-a58a-31d0cf333035" width=1000>
 </div>
 <br>

**Key Components**

| Component | Purpose | Best Practices |
|-----------|---------|----------------|
| Tokenizer | Text → Model Input | Use same parameters as training |
| Model | Prediction | Always use `eval()` mode for inference |
| Label Encoder | Decode predictions | Validate against known classes |


## Zipping the Trained Model Directory for Export or Deployment
<div>
 <img src="https://github.com/user-attachments/assets/af691b71-009c-4369-b4a4-f23da7fd111b" width=1000>
 </div>
 <br>

**Purpose**
Creates a portable archive containing:
- Trained DistilBERT model weights
- Tokenizer vocabulary
- Label encoder
- Training logs and checkpoints

**Archive Contents Structure**
```
results.zip
├── best_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── vocab.txt
│   ├── label_encoder.pkl
│   └── training_args.bin
├── logs/
│   └── training_metrics.json
└── checkpoints/
    ├── checkpoint-1000/
    └── checkpoint-2000/
```

## Checking the Details of the Compressed Model Archive
<div>
 <img src="https://github.com/user-attachments/assets/fa8edc5a-e1d7-41e9-b349-d3dca76c989d" width=1000>
 </div>
 <br>

**Purpose**
Verifies the existence and basic properties of your zipped model artifacts before distribution or deployment.

**Expected Output Format**
```
-rw-r--r-- 1 root root 1.2G May 20 14:30 /content/results.zip
```

**Output Interpretation**
| Field | Meaning | Importance |
|-------|---------|------------|
| File Permissions | `-rw-r--r--` | Ensures proper access control |
| Owner | `root` | Should match your user context |
| Size | `1.2G` | Verifies complete compression |
| Timestamp | `May 20 14:30` | Confirms recent version |











  
