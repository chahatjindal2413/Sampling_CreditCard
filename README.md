# Credit Card Fraud Detection using Sampling Techniques

## 📌 Objective
The objective of this assignment is to analyze the impact of different sampling techniques on machine learning model performance when dealing with an **imbalanced dataset**.

In real-world problems such as fraud detection, fraudulent transactions are extremely rare compared to genuine ones. This imbalance causes machine learning models to become biased toward the majority class.

This project evaluates how balancing the dataset improves classification performance.

---

## 📂 Dataset
Dataset used: **Credit Card Fraud Dataset**

The dataset contains transaction records classified as:

- **0 → Normal Transaction**
- **1 → Fraudulent Transaction**

The dataset is highly imbalanced where normal transactions greatly outnumber fraudulent ones.

---

## ⚙️ Methodology

### Step 1 — Data Preprocessing
1. Load dataset
2. Check class imbalance
3. Split dataset into training and testing sets using stratified sampling

---

### Step 2 — Sampling Techniques Applied
Five sampling techniques were used to balance the dataset:

| Sampling | Technique |
|--------|------|
| Sampling 1 | Random Under Sampling |
| Sampling 2 | Random Over Sampling |
| Sampling 3 | SMOTE |
| Sampling 4 | ADASYN |
| Sampling 5 | SMOTEENN (Hybrid) |

---

### Step 3 — Machine Learning Models
Five machine learning models were trained on each sampled dataset:

| Model | Algorithm |
|------|------|
| M1 | Logistic Regression |
| M2 | K-Nearest Neighbors |
| M3 | Decision Tree |
| M4 | Random Forest |
| M5 | Support Vector Machine |

Total Experiments = **5 Sampling × 5 Models = 25 Experiments**

---

## 📊 Evaluation Metric
Models were evaluated using:

**Accuracy Score**

Accuracy was calculated on the untouched test dataset to ensure unbiased evaluation.

---

## 📈 Results
The accuracy results for each combination were stored in:

```
results/accuracy_table.csv
```

---

## 📊 Performance Heatmap

The following heatmap shows accuracy comparison of all models across different sampling techniques:
<img width="1098" height="801" alt="image" src="https://github.com/user-attachments/assets/cdd88ce6-8794-4114-8d82-665c94be800b" />


### Interpretation
- Darker color indicates higher accuracy
- Each row represents a Machine Learning model
- Each column represents a sampling technique

Observations:
- Synthetic sampling methods (SMOTE, ADASYN, SMOTEENN) performed better than random sampling
- Undersampling showed unstable performance due to information loss
- Random Forest and SVM achieved comparatively higher accuracy
- Hybrid sampling produced consistent results across models

---

## 🧠 Observations & Discussion

1. The original dataset was highly imbalanced, causing models to predict majority class only.
2. Random undersampling removed useful data leading to performance drop.
3. Random oversampling improved performance but risked overfitting due to duplicate samples.
4. SMOTE and ADASYN performed significantly better as they generated synthetic minority samples.
5. Hybrid sampling (SMOTEENN) provided balanced and cleaner data.
6. Random Forest showed stable performance due to ensemble learning reducing variance.

---

## 🏆 Conclusion
Balancing the dataset significantly improves machine learning performance on imbalanced classification problems.

Key findings:

- Oversampling techniques outperform undersampling
- Synthetic data generation works better than duplication
- Hybrid methods provide stable results
- Ensemble models perform most reliably

Therefore, selecting an appropriate sampling technique is crucial for fraud detection systems.

---

## 📌 Key Learning

This experiment demonstrates that machine learning performance depends not only on the model but also on data distribution.

Imbalanced datasets bias models toward majority class prediction.  
Sampling techniques help the model learn minority class patterns, which is critical in real-world applications like:

- Fraud detection
- Disease prediction
- Spam detection
- Intrusion detection systems

Thus, preprocessing is as important as model selection in machine learning pipelines.

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn
- Google Colab

---

## ▶️ How to Run
1. Open Google Colab
2. Upload `Creditcard_data.csv`
3. Run `sampling_models.ipynb`
4. Results will be generated automatically



---

## 📚 Learning Outcome
This project highlights the importance of handling class imbalance in machine learning and demonstrates that proper preprocessing techniques can significantly improve model reliability in real-world classification problems.
