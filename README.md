# 🚦 Accident Severity Prediction (AI/ML)

Road accidents aren’t all the same—some are minor, some are life-threatening. This project uses AI/ML to predict how severe an accident outcome could be, using real accident data. I built and evaluated multiple classifiers, with a key focus on Person Injury Severity, where I applied SMOTE + Random Forest to handle imbalanced classes and deliver more reliable predictions.

---

## 🔥 Project Highlights
- ✅ Built an end-to-end ML pipeline for **severity classification**
- ✅ Focused on **Person Injury Severity (P_ISEV)** using:
  - **Random Forest (Ensemble Learning)**
  - **SMOTE** to handle class imbalance
- ✅ Tried ensemble strategies like **Voting / Stacking** (for comparison)
- ✅ Evaluated with **Accuracy, Confusion Matrix, Classification Report**

---

## 🧠 Problem Statement
Accidents vary from minor to fatal. This project predicts severity levels from accident-related features so we can:
- understand risk patterns,
- compare models,
- and identify the best-performing approach for severity classification.

---

## 🧰 Tech Stack
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Imbalanced-learn (SMOTE)**
- **Matplotlib / Seaborn (visualizations)**
- **Jupyter Notebook**

---

## ⚙️ ML Workflow
1. Data loading + basic cleaning  
2. Feature preparation & encoding (`get_dummies`)  
3. Train-test split (stratified)  
4. Model training:
   - Random Forest (core contribution)
   - Ensemble methods (Voting / Stacking)
5. Handling imbalance:
   - **SMOTE** for P_ISEV
6. Evaluation:
   - Accuracy
   - Confusion matrix
   - Classification report

---

## 🧪 Models Used
### For Person Injury Severity (P_ISEV)
- Random Forest ✅ (with SMOTE)
- Voting Classifier
- Stacking Classifier

### For Collision Severity (C_SEV)
- Random Forest
- Other classifiers for comparison

---

## 📊 Results (Snapshot)
> Your project compares multiple models — best result may vary depending on feature selection & sampling.

**Example from evaluation table:**
- **Best P_ISEV accuracy:** ~71.79% (Stacking)
- **Random Forest baseline for P_ISEV:** ~65.50%
- **SMOTE + Ensemble improvement observed** in experiments
## 📊 Output / Results

### Results Comparison Table
![Results Table](output.jpg)

*(Exact results are available inside the notebook outputs.)*

---

## 👨‍💻 My Contribution
- Implemented **Ensemble Learning + SMOTE + Random Forest** for **Person Injury Severity (P_ISEV)**
- Built the training + evaluation pipeline (accuracy, confusion matrix, classification report)
- Compared performance with ensemble methods to validate improvements

---

## ▶️ How to Run
### 1) Clone the repository
```bash
git clone https://github.com/Anuz-bit/accident-severity-prediction.git
cd accident-severity-prediction
```
---
## 📌 Dataset & Credits
This project uses an accident dataset sourced from **Kaggle**. Full credit goes to the original dataset publisher and Kaggle for hosting the data.  
If you are the dataset owner and would like the credit format changed or the data removed from this repository, feel free to reach out.

