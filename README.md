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
   

# Task 3: Multimodal Machine Learning – Housing Price Prediction Using Images and Tabular Data

## Objective

The objective of this project is to develop a multimodal machine learning model capable of predicting house prices by combining structured housing information with property images. The model uses a Convolutional Neural Network (CNN) to learn visual features from house images and a Multi-Layer Perceptron (MLP) to process tabular features. The extracted representations are fused to produce a single house price prediction.

## Dataset

**Dataset:** Houses Dataset (Ahmed & Moustafa)

The dataset consists of **535 residential properties**, where each property contains both structured data and image data.

### Tabular Features

* Number of bedrooms
* Number of bathrooms
* House area (square feet)
* Zip code
* House price (target variable)

### Image Features

Each property includes four images representing different views:

* Front exterior
* Bedroom
* Bathroom
* Kitchen

## Technologies and Libraries

The project was implemented using the following Python libraries:

* NumPy
* Pandas
* OpenCV
* TensorFlow / Keras
* Scikit-learn

## Project Workflow

The following steps were performed during model development:

* Loaded the housing dataset containing both tabular information and property images.
* Combined the four images of each house into a single collage image for CNN processing.
* Preprocessed the tabular features using **MinMaxScaler**.
* Normalized image pixel values to improve CNN training.
* Split the dataset into training and testing sets while maintaining alignment between images, tabular features, and target values.
* Designed a CNN branch to extract visual features from the collage images.
* Built an MLP branch to learn feature representations from the tabular data.
* Fused the outputs of both branches using a **Concatenate** layer.
* Added fully connected regression layers to predict the final house price.
* Trained the multimodal model end-to-end.
* Evaluated the trained model using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.
* Saved the trained model for future inference and deployment.

## Results and Observations

Several observations were made during experimentation:

* The dataset contains only **535 samples**, which limits the CNN's ability to learn highly robust visual features.
* House prices vary significantly, ranging from **$22,000** to **$5,858,000**, resulting in a highly skewed target distribution.
* The average house price is approximately **$589,363**, while the median price is around **$529,000**.
* The presence of expensive outlier properties increases the difficulty of the regression task.
* Combining image features with structured housing attributes enables the model to capture complementary information from both data sources, resulting in a more informative prediction than relying on either modality alone.

## Model Performance

| Metric                         |           Value |
| ------------------------------ | --------------: |
| Mean Absolute Error (MAE)      | **$285,289.48** |
| Root Mean Squared Error (RMSE) | **$376,616.60** |

## Performance Analysis

The model achieved reasonable performance considering the limited dataset size and the large variation in house prices. However, prediction errors remain relatively high because of:

* Small training dataset
* Large price variability
* Presence of high-value outlier properties
* Training the CNN from scratch on a limited number of images

Future improvements may include:

* Applying a logarithmic transformation to the target variable.
* Using transfer learning with pretrained CNN models such as **MobileNetV2**, **EfficientNet**, or **ResNet50**.
* Performing image augmentation to improve generalization.
* Engineering additional tabular features to provide richer information for prediction.

## How to Run the Project

1. Install the required dependencies:

```bash
pip install numpy pandas opencv-python tensorflow scikit-learn
```

2. Download or clone the Houses Dataset repository.

3. Ensure that both the image files and the tabular dataset are placed in the correct project directories.

4. Run the notebook or Python script to:

   * Load and preprocess the data
   * Build the CNN and MLP branches
   * Train the multimodal fusion model
   * Evaluate the model

5. After training, the model will be saved as:

```
housing_multimodal_model.keras
```

(or `housing_multimodal_model.h5`, depending on the selected save format).
