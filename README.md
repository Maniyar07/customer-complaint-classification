# Customer Complaint Classification System

A Streamlit web app that predicts the category of a customer financial complaint using an NLP machine learning pipeline.

## Project Summary

This project classifies customer complaint text into financial complaint categories such as:

- Credit card or prepaid card
- Checking or savings account
- Mortgage
- Debt collection
- Money transfer, virtual currency, or money service
- Student loan
- Vehicle loan or lease
- Credit reporting, credit repair services, or other personal consumer reports

The final model uses TF-IDF feature extraction and Logistic Regression.

## Final Model Performance

| Metric | Value |
|---|---:|
| Accuracy | 81.98% |
| Macro F1 | 74.51% |
| Weighted F1 | 82.27% |
| Number of Classes | 8 |
| Training Rows Used | 20,840 |

## Recommended Project Structure

```text
NLP_PROJECT/
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── models/
│   ├── best_complaint_classifier_pipeline.pkl
│   └── best_label_encoder.pkl
├── reports/
│   ├── 03_best_model_metrics.json
│   └── 03_best_classification_report.txt
├── assets/
│   ├── top_15_categories.png
│   └── complaint_word_count_distribution.png
├── notebooks/
│   ├── 01_EDA_AND_DATA_UNDERSTANDING.ipynb
│   ├── 02_TEXT_PREPROCESSING_AND_FEATURE_ENGINEERING.ipynb
│   └── 03_MODEL_TRAINING_EVALUATION_AND_EXPORT.ipynb
├── data/
└── outputs/
```

For deployment, the important files are:

```text
app.py
requirements.txt
Dockerfile
models/best_complaint_classifier_pipeline.pkl
models/best_label_encoder.pkl
reports/03_best_model_metrics.json
reports/03_best_classification_report.txt
assets/top_15_categories.png
assets/complaint_word_count_distribution.png
```

## Run Locally

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Run with Docker

Build the Docker image:

```bash
docker build -t complaint-classifier-app .
```

Run the container:

```bash
docker run -p 8501:8501 complaint-classifier-app
```

Open:

```text
http://localhost:8501
```

## Important Notes

- Keep the Streamlit code in `app.py` at the project root for simple deployment.
- Keep notebooks inside the `notebooks/` folder.
- Keep model files inside the `models/` folder.
- Keep reports and charts inside `reports/` and `assets/`.
- Do not upload very large raw datasets to deployment unless the app needs them.
- Do not delete `.pkl` model files because the app needs them for prediction.

## Files Safe to Ignore or Delete Before Deployment

Usually safe to remove from deployment:

```text
.ipynb_checkpoints/
__pycache__/
*-checkpoint.ipynb
old duplicate CSV files
raw Excel dataset, if the app does not read it
```

Do not remove:

```text
app.py
requirements.txt
Dockerfile
models/best_complaint_classifier_pipeline.pkl
models/best_label_encoder.pkl
```
