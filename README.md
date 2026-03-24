# 💳 AI Credit Default Prediction System

An end-to-end machine learning project that predicts whether a customer will default on credit payments.

##  Features

* Ensemble model (LightGBM + Random Forest + Logistic Regression)
* Explainable AI using SHAP
* Interactive UI using Streamlit

## 📁 Project Structure

```
credit-risk-project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
├── data/
├── assets/
```

## ⚙️ How to Run

### 1. Clone repo

```
git clone https://github.com/your-username/credit-risk-project.git
cd credit-risk-project
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run app

```
streamlit run app.py
```

##  Tech Stack

* Python
* Streamlit
* Scikit-learn
* LightGBM
* SHAP

##  End-to-End Project Workflow

This project was built as a complete real-world machine learning system, covering everything from raw data preprocessing to deployment with explainable AI.

---

###  1. Data Understanding & Cleaning

* Started with the **Credit Card Default dataset**
* Checked for:

  * Missing values
  * Data inconsistencies
  * Incorrect data types
* Filled missing values where necessary and ensured all features were usable for modeling

---

###  2. Data Preprocessing

* Converted categorical features into numerical format
* Standardized feature naming for consistency
* Removed redundant or irrelevant columns
* Ensured data was model-ready

---

###  3. Outlier Handling

* Identified outliers using statistical methods and domain understanding
* Instead of blindly removing them:

  * Preserved meaningful extreme values (important in financial risk)
  * Avoided over-cleaning to retain real-world variability

---

###  4. Feature Engineering

Created new powerful features to improve model performance:

* **Payment Ratio** = Total Payment / Total Bill
* **Utilization Ratio** = Total Bill / Credit Limit
* **Payment Consistency** = Total Payment / Credit Limit

These features helped capture customer behavior more effectively than raw data.

---

###  5. Model Training

Trained multiple models to compare performance:

* Logistic Regression (baseline)
* Random Forest (robust ensemble)
* LightGBM (high-performance gradient boosting)

---

###  6. Ensemble Learning

Instead of relying on a single model:

* Combined predictions using weighted averaging:

  * LightGBM → 70%
  * Random Forest → 20%
  * Logistic Regression → 10%

This improved accuracy and generalization.

---

###  7. Pipeline Design

* Used preprocessing + scaling pipelines for consistency
* Ensured the same transformations were applied during training and inference
* Saved models and scalers using `joblib` for deployment

---

###  8. Explainable AI (SHAP)

* Used SHAP (SHapley Additive Explanations) to:

  * Explain individual predictions
  * Show feature impact visually
* Added:

  * Waterfall plots
  * Feature importance charts

This makes the model transparent and trustworthy.

---

###  9. Streamlit Deployment

* Built an interactive UI using Streamlit
* Designed user-friendly input forms with explanations
* Integrated:

  * Real-time prediction
  * Risk classification
  * SHAP visualizations

---

###  10. Debugging & Real-World Challenges

During development, several practical issues were encountered:

* ❌ File path errors (`\` vs `/`) → fixed using proper path handling
* ❌ Library issues (SHAP not installed) → resolved via dependency management
* ❌ Conda environment issues in VS Code → fixed using interpreter setup
* ❌ Data mismatch during inference → resolved using consistent pipelines

---

###  11. Final Outcome

* A fully working **AI-powered credit risk prediction system**
* Combines:

  * Machine Learning
  * Explainable AI
  * Web deployment

---

## 💡 Key Learnings

* Feature engineering can outperform complex models
* Ensemble models improve robustness
* Explainability is crucial in financial applications
* Deployment requires handling real-world system issues, not just modeling


## 👨‍💻 Author

Pritish Kumar Singh
##  Connect With Me

* 📧 Email: [pritishsinghprf@gmail.com](mailto:pritishsinghprf@gmail.com)
* 💼 LinkedIn: www.linkedin.com/in/pritish1298

---

