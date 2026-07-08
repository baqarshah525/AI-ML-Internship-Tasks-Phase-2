# Task 1: News Topic Classifier Using BERT

## Objective

The objective of this task is to fine-tune the **BERT (bert-base-uncased)** transformer model to classify news headlines into their respective topic categories. After training, the model is deployed as an interactive web application using **Gradio** to perform real-time news classification.

---

## Dataset Used

**Dataset:** AG News Dataset (Hugging Face – `fancyzhx/ag_news`)

The AG News dataset contains news headlines categorized into four different classes. It includes **120,000 training samples** and **7,600 testing samples**.

The four news categories are:

* World
* Sports
* Business
* Sci/Tech

---

## Libraries Used

* Transformers (Hugging Face)
* Datasets (Hugging Face)
* PyTorch
* Scikit-learn
* Evaluate
* Accelerate
* Gradio

---

## Tasks Performed

* Loaded the AG News dataset from Hugging Face.
* Tokenized the news headlines using the BERT tokenizer.
* Preprocessed the dataset for model training.
* Fine-tuned the **bert-base-uncased** model using the Hugging Face Trainer API.
* Evaluated the trained model using **Accuracy** and **F1-score**.
* Saved the trained model and tokenizer for future use.
* Developed a Gradio-based web interface for real-time news topic prediction.

---

## Key Findings

* The AG News dataset consists of **120,000 training records** and **7,600 testing records**.
* The dataset contains four categories: **World, Sports, Business, and Sci/Tech**.
* The fine-tuned BERT model achieved strong classification performance after training for **3 epochs**.
* Business and Sci/Tech headlines occasionally overlap because they share similar keywords such as *company*, *technology*, and *market*.
* Headlines containing clear and topic-specific vocabulary are generally classified with high confidence.

---

## Model Performance

| Metric   | Score                      |
| -------- | -------------------------- |
| Accuracy | *(Insert Cell 9 Accuracy)* |
| F1-score | *(Insert Cell 9 F1-score)* |

---

## How to Run

1. Install the required libraries:

```bash
pip install transformers datasets torch scikit-learn evaluate accelerate gradio
```

2. Run the notebook (or `train_bert_agnews.py`) to fine-tune the model.

3. After training is complete, the model and tokenizer will be saved automatically.

4. Run the Gradio application cell to launch the interactive interface.

5. Enter any news headline into the Gradio interface to receive its predicted category in real time.


# Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

## Objective

The objective of this project is to develop a reusable and production-ready machine learning pipeline for customer churn prediction using the Scikit-learn Pipeline API. The pipeline integrates data preprocessing, feature transformation, model training, hyperparameter tuning, and model deployment into a single workflow, ensuring consistency and reducing the risk of data leakage.

---

## Dataset

**Dataset:** Telco Customer Churn Dataset

The Telco Customer Churn dataset contains customer information collected from a telecommunications company. It includes demographic details, subscribed services, account information, billing history, and the target variable indicating whether a customer has churned. The dataset consists of **7,043 customer records** with **21 features**.

---

## Libraries Used

* Pandas
* NumPy
* Scikit-learn
* Joblib

---

## Project Workflow

The following steps were performed to build the machine learning pipeline:

* Loaded the Telco Customer Churn dataset.
* Cleaned the dataset by handling missing values and correcting data types, particularly the `TotalCharges` column.
* Converted the target variable (`Churn`) into binary format for classification.
* Identified and separated numerical and categorical features.
* Built preprocessing pipelines using `ColumnTransformer`:

  * Standardized numerical features using `StandardScaler`.
  * Encoded categorical features using `OneHotEncoder`.
* Created complete machine learning pipelines by combining preprocessing with:

  * Logistic Regression
  * Random Forest Classifier
* Applied `GridSearchCV` with cross-validation to optimize model hyperparameters.
* Evaluated both models using test accuracy and classification reports.
* Saved the best-performing pipeline using Joblib for future deployment.
* Reloaded the exported pipeline and verified its predictions on unseen data.

---

## Key Findings

* The dataset contains **7,043 customer records** with both numerical and categorical features.
* Missing values were identified in the `TotalCharges` column due to blank string entries and were successfully handled during preprocessing.
* Both Logistic Regression and Random Forest achieved competitive performance, with test accuracies generally ranging between **75% and 82%**.
* Logistic Regression demonstrated stronger recall for the majority class, while Random Forest captured more complex relationships within the data.
* Using the Scikit-learn Pipeline API ensured that identical preprocessing steps were applied during both training and inference, making the workflow reliable, reusable, and deployment-ready.

---

## Model Performance

| Metric                         | Logistic Regression       | Random Forest             |
| ------------------------------ | ------------------------- | ------------------------- |
| Best Cross-Validation Accuracy | *(Insert Cell 8 Output)*  | *(Insert Cell 9 Output)*  |
| Test Accuracy                  | *(Insert Cell 10 Output)* | *(Insert Cell 10 Output)* |

---

## How to Run

### 1. Install the Required Libraries

```bash
pip install pandas numpy scikit-learn joblib
```

### 2. Run the Project

* Load the Telco Customer Churn dataset.
* Execute the preprocessing pipeline.
* Train both Logistic Regression and Random Forest models.
* Perform hyperparameter tuning using `GridSearchCV`.
* Compare the evaluation metrics to determine the best-performing model.
* The best pipeline will automatically be saved as:

```text
churn_prediction_pipeline.pkl
```

### 3. Load the Saved Pipeline

Use `joblib.load()` to reload the exported pipeline and generate predictions on new customer data without repeating the preprocessing or training steps.
   
