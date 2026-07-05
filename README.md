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
