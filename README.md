
# 📩 Customer Complaint Classification System

A machine learning-based NLP project that classifies customer complaint text into the correct complaint category.  
The project uses **TF-IDF Vectorization** and **Logistic Regression** for text classification.

This project includes:

- Data understanding and EDA
- Text preprocessing and feature engineering
- Model training and evaluation
- Flask backend API
- Streamlit frontend UI
- Model performance reports and visualizations

---

## 🚀 Project Overview

Customer complaints are usually written in natural language. Manually reading and assigning each complaint to a category is time-consuming.

This project solves that problem by automatically predicting the complaint category from complaint text.

### Example Input

Money was debited from my account without my permission.

### Example Output

Checking or savings account

## 🧠 Machine Learning Approach

The project uses a traditional NLP machine learning pipeline:

Complaint Text
     ↓
Text Cleaning
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression Model
     ↓
Predicted Complaint Category

## 📊 Model Performance

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 81.87% |
| Macro F1          | 73.78% |
| Weighted F1       | 82.23% |
| Training Rows     | 20,874 |
| Number of Classes |      8 |


## 🗂️ Project Structure

NLP_PROJECT/
│
├── app.py
├── streamlit_app.py
├── Dockerfile
├── README.md
├── requirements.txt
│
├── assets/
│   ├── complaint_word_count_distribution.png
│   └── top_15_categories.png
│
├── data/
│   └── complaints_copy.xlsx
│
├── models/
│   ├── best_complaint_classifier_pipeline.pkl
│   └── best_label_encoder.pkl
│
├── notebooks/
│   ├── 01_EDA_AND_DATA_UNDERSTANDING.ipynb
│   ├── 02_TEXT_PREPROCESSING_AND_FEATURE_ENGINEERING.ipynb
│   └── 03_MODEL_TRAINING_EVALUATION.ipynb
│
├── outputs/
│   ├── 01_nlp_base_complaints.csv
│   └── 02_processed_complaints.csv
│
└── reports/
    ├── 03_best_classification_report.csv
    ├── 03_best_classification_report.txt
    ├── 03_best_model_metrics.json
    ├── 03_data_feature_experiment_results.csv
    ├── 03_misclassified_samples.csv
    ├── complaint_word_count_distribution.png
    ├── missing_values_report.csv
    └── top_15_categories.png

## 📁 Folder Explanation

### `data/`

Contains the original raw dataset.

### `notebooks/`

Contains the complete step-by-step project workflow:

1. **EDA and Data Understanding**
2. **Text Preprocessing and Feature Engineering**
3. **Model Training and Evaluation**

### `outputs/`

Contains cleaned and processed datasets generated from notebooks.

### `models/`

Contains saved trained model files:

* `best_complaint_classifier_pipeline.pkl`
* `best_label_encoder.pkl`

### `reports/`

Contains model performance reports, classification report, experiment results, misclassified samples, and EDA chart images.

### `assets/`

Contains visualization images used in the UI.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* TF-IDF Vectorizer
* Logistic Regression
* Flask
* Streamlit
* Joblib
* Matplotlib
* Docker

---

## 🔥 Features

* Predicts customer complaint category from text
* Shows model performance metrics
* Displays prediction confidence scores
* Provides EDA visualizations
* Shows classification report
* Includes Flask API backend
* Includes Streamlit frontend UI
* Organized project structure for resume and GitHub

---

## 🧪 Model Details

### Text Feature

clean_text

### Vectorization

TF-IDF Vectorizer

### Model

Logistic Regression

### Best Configuration

Text Version: clean_text
Max Features: 30000
Min Samples Per Class: 50

---

## ▶️ How to Run Locally

### Step 1: Clone the Repository

git clone <your-github-repository-link>
cd NLP_PROJECT

---

### Step 2: Create Virtual Environment

python -m venv .venv

Activate it:

#### Windows

.venv\Scripts\activate

#### Mac/Linux

source .venv/bin/activate

---

### Step 3: Install Requirements

pip install -r requirements.txt

---

## 🚀 Run Flask Backend

Open terminal 1:

python app.py

Flask API will run at:

http://127.0.0.1:5000

Useful API endpoints:

GET  /
GET  /health
GET  /metrics
POST /predict
---

## 🎨 Run Streamlit Frontend

Open terminal 2:

streamlit run streamlit_app.py

Streamlit app will run at:

http://localhost:8501

---

## 🔌 API Example

### Request

{
  "complaint_text": "My credit card was charged without my permission."
}

### Response

{
  "status": "success",
  "predicted_category": "Credit card or prepaid card",
  "confidence_scores": []
}

---

## 📊 Reports Generated

The project generates the following reports:

03_best_classification_report.csv
03_best_classification_report.txt
03_best_model_metrics.json
03_data_feature_experiment_results.csv
03_misclassified_samples.csv
missing_values_report.csv

These reports help in understanding:

* Model performance
* Class-wise precision, recall, and F1-score
* Best experiment configuration
* Misclassified complaint examples
* Missing values in dataset

---

## 📈 Visualizations

The project includes EDA charts such as:

* Top 15 complaint categories
* Complaint word count distribution

These charts are used inside the Streamlit UI.

---

## 🐳 Docker Support

Build Docker image:

docker build -t customer-complaint-classifier .

Run Docker container:

docker run -p 8501:8501 customer-complaint-classifier

---

## 🌐 Deployment

The app can be deployed using:

* Render
* Streamlit Community Cloud
* Railway
* Docker-based deployment

For simple deployment, use only Streamlit.

For professional deployment, use:

Streamlit Frontend → Flask API Backend → ML Model

## 📌 Important Notes

* Do not use absolute local paths like `C:\Users\...` in deployment.
* Always use project-relative paths.
* Keep model files inside the `models/` folder.
* Keep reports and chart images inside the `reports/` or `assets/` folder.
* Restart Flask and Streamlit after updating model or backend code.

---

## 👨‍💻 Author

**Maniyar Sohel**

Project: Customer Complaint Classification System
Domain: Natural Language Processing and Machine Learning

---

## ✅ Final Result

This project demonstrates an end-to-end NLP machine learning workflow, including data preprocessing, model training, evaluation, backend API development, and frontend UI deployment.

```

One small suggestion: keep chart images in only one place. Since your app currently uses reports/assets depending on code, choose either `reports/` or `assets/`. For cleaner structure, keep UI images in `assets/` and reports like CSV/JSON/TXT in `reports/`.
```
