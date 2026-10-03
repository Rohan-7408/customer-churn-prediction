
# 📊 Customer Churn Prediction

A Machine Learning web application that predicts whether a customer is likely to leave a service using **Logistic Regression** and a Scikit-learn preprocessing pipeline.

## 🚀 Live Deployment

**[Click here to use the Customer Churn Prediction App](https://customer-churn-prediction-avf9.onrender.com/)**

## 💻 GitHub Repository

[View Source Code on GitHub](https://github.com/Rohan-7408/customer-churn-prediction)

---

## 📌 Project Overview

Customer churn occurs when customers discontinue their relationship with a company or stop using its services.

This project uses machine learning to classify customers into two categories:

- **Churn:** Customer is predicted to leave.
- **No Churn:** Customer is predicted to stay.

The application provides an interactive interface for entering customer information and generating churn predictions.

## 🎯 Objectives

- Predict customer churn using machine learning.
- Preprocess customer data before model training.
- Handle class imbalance using balanced class weights.
- Build a reusable machine learning pipeline.
- Deploy the application as a web app.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Algorithm | Logistic Regression |
| Data Processing | Pandas, NumPy |
| Web Framework | Streamlit |
| Deployment | Render |
| Development | Google Colab, GitHub |

## 🤖 Machine Learning Algorithm

The project uses **Logistic Regression**, a supervised machine learning algorithm for binary classification.

The model is configured with:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
```

- `max_iter=1000`: Sets the maximum number of optimization iterations.
- `class_weight="balanced"`: Adjusts class weights to account for class imbalance.
- `random_state=42`: Supports reproducibility where randomness is involved.

## 🔄 Model Pipeline

The project combines preprocessing and classification into a Scikit-learn pipeline.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)
```

### Workflow

1. Customer data is collected.
2. Data is preprocessed using the configured preprocessor.
3. The Logistic Regression model processes the transformed features.
4. The model generates a churn prediction.
5. The application displays the result.

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── requirements.txt
├── README.md
└── ...
```

## ⚙️ Run Locally

Clone the repository:

```bash
git clone https://github.com/Rohan-7408/customer-churn-prediction.git
```

Move into the project directory:

```bash
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 🌐 Deployment on Render

The application is deployed using Render.

**Build Command:**

```bash
pip install -r requirements.txt
```

**Start Command:**

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### 🔗 Application Links

- **Live App:** https://customer-churn-prediction-avf9.onrender.com/
- **GitHub:** https://github.com/Rohan-7408/customer-churn-prediction

## 🔮 Future Improvements

- Compare additional classification algorithms.
- Perform hyperparameter tuning.
- Add confusion matrix and classification metrics.
- Improve feature engineering.
- Display prediction probabilities.
- Evaluate the model on additional datasets.

## 👨‍💻 Author

**Akhand Pratap Vishwakarma**  
B.Tech – Computer Science & Engineering

Interested in Data Analytics, Data Science, Machine Learning, and Software Development.

---

*Developed for educational and internship purposes. Predictions are model-generated estimates and are not guaranteed business outcomes.*
