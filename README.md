
# ❤️ 7072CEM: Machine Learning – Heart Disease Detection

This repository contains a complete machine learning pipeline for heart disease prediction using multiple classification algorithms. The project evaluates and compares the performance of four different models—Support Vector Machine (SVM), Logistic Regression (LR), Random Forest (RF), and K-Nearest Neighbors (KNN)—on a public dataset using key clinical metrics.

---

## 📁 Repository Structure

| File                            | Description                                                      |
|---------------------------------|------------------------------------------------------------------|
| `Machine_Learning_Module.ipynb` | Main Jupyter Notebook implementing the entire ML pipeline        |
| `README.md`                     | Overview and documentation (this file)                          |

---

## 🎯 Objective

To identify the most effective machine learning algorithm for detecting heart disease based on medical and clinical features, and to compare their performance using metrics such as accuracy, precision, recall, and F1-score.

---

## 📊 Dataset

- **Source**: UCI Machine Learning Repository – [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Size**: 303 records with 13 features + 1 binary target
- **Features**: Age, sex, chest pain type, cholesterol, blood pressure, etc.
- **Target**: Presence (1) or absence (0) of heart disease
- **Preprocessing**: 
  - Invalid/missing values cleaned
  - Categorical variables one-hot encoded
  - Normalization skipped due to Gaussian-like distributions
  - 80/20 stratified train-test split

---

## 🤖 Models Evaluated

| Algorithm          | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 86.49%   | 82%       | 91%    | 0.86     | 0.92    |
| Random Forest      | 83.78%   | **83%**   | 83%    | 0.83     | 0.92    |
| SVM                | 64.86%   | 58%       | 89%    | 0.70     | 0.80    |
| K-Nearest Neighbors| 55.41%   | 55%       | 31%    | 0.40     | 0.60    |

> **Conclusion**: Logistic Regression performed best overall with high accuracy and recall, making it a suitable model for clinical diagnostics.

---

## 📦 Tools & Technologies

- Python
- Scikit-learn
- Pandas, NumPy
- Seaborn & Matplotlib for data visualization
- Google Colab

---

## 📌 Highlights

- Logistic Regression achieved the best balance of metrics.
- Random Forest offered strong precision and could support screening tools.
- SVM and KNN underperformed on this dataset.
- Visual insights through ROC curves and confusion matrices supported analysis.

---

## 🚀 Future Improvements

- Try ensemble techniques and neural networks
- Apply feature selection and dimensionality reduction
- Expand to time-series cardiac data or multi-class scenarios
- Integrate model explainability tools (e.g., SHAP)

---

## 🧑‍💻 Author

**Itorobong Akpan**  
MSc Data Science & Computational Intelligence  
Coventry University, UK  
📧 akpani4@uni.coventry.ac.uk  
🔗 [GitHub Profile](https://github.com/akpanitorobong)

---
