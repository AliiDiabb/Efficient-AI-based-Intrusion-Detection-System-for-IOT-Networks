# 🎯 Running 15-class classification experiment

## 📂 Dataset Information
- Dataset loaded successfully. Shape: **(156986, 51)**
- Total samples: **156986**
- Total features: **51**
- Missing values: **0**
- Training set: **125588 samples**
- Testing set: **31398 samples**

- Using **Attack_type** for 15-class classification (all attack types)
- Features shape: **(156986, 47)**

---

## 📊 Target distribution (15-class)

0 10195
1 10561
2 14090
3 10247
4 14498
5 1001
6 400
7 24301
8 9989
9 10071
10 10925
11 10311
12 10269
13 10076
14 10052
Name: Attack_type, dtype: int64


- Class imbalance ratio: **0.016**
- ⚠️ Severe class imbalance detected - this may lead to overfitting!

---

# 🚀 Training LightGBM Classifier (15-class)

### 🔍 Phase 1: Quick parameter screening
- Fitting 3 folds for each of 20 candidates, totalling 60 fits

### 🔍 Enhanced Cross-Validation
- Cross-validation Accuracy: **0.9536 (+/- 0.0026)**
- Cross-validation Precision: **0.9543 (+/- 0.0030)**
- Cross-validation Recall: **0.9427 (+/- 0.0054)**
- Cross-validation F1-Score: **0.9474 (+/- 0.0041)**

**Overfitting Analysis**
- Training Score: **0.9576**
- Validation Score: **0.9536**
- Overfitting Gap: **0.0040**
- CV Variance: **0.000002**
- ✅ No significant overfitting detected

### 📊 FINAL RESULTS
- Training completed in **1086.86 seconds**
- Prediction completed in **0.74 seconds**
- Test Accuracy: **0.9525**
- Training Accuracy: **0.9567**
- Final Overfitting Gap: **0.0041**

### Classification Report (15-class)

          precision    recall  f1-score   support

       0       1.00      0.96      0.98      2039
       1       0.83      0.89      0.86      2112
       2       1.00      1.00      1.00      2818
       3       1.00      1.00      1.00      2050
       4       1.00      1.00      1.00      2900
       5       0.98      0.82      0.90       200
       6       1.00      1.00      1.00        80
       7       1.00      1.00      1.00      4860
       8       0.95      0.82      0.88      1998
       9       0.92      0.96      0.94      2014
      10       0.89      0.95      0.92      2185
      11       0.89      0.90      0.89      2062
      12       0.96      0.90      0.93      2054
      13       0.99      0.97      0.98      2015
      14       0.87      0.95      0.91      2011

accuracy                           0.95     31398
macro avg     0.95       0.94      0.95     31398
weighted avg  0.95       0.95      0.95     31398


---

## 📊 Target distribution (6-class)

0 24301
1 49396
2 41378
3 21148
4 20363
5 400
Name: 6_Attack, dtype: int64



- Class imbalance ratio: **0.008**
- ⚠️ Severe class imbalance detected - this may lead to overfitting!

---

# 🚀 Training LightGBM Classifier (6-class)

### 🔍 Enhanced Cross-Validation
- Cross-validation Accuracy: **0.9566 (+/- 0.0023)**
- Cross-validation Precision: **0.9600 (+/- 0.0052)**
- Cross-validation Recall: **0.9610 (+/- 0.0048)**
- Cross-validation F1-Score: **0.9605 (+/- 0.0048)**

**Overfitting Analysis**
- Training Score: **0.9583**
- Validation Score: **0.9566**
- Overfitting Gap: **0.0017**
- CV Variance: **0.000001**
- ✅ No significant overfitting detected

### 📊 FINAL RESULTS
- Training completed in **719.80 seconds**
- Prediction completed in **0.26 seconds**
- Test Accuracy: **0.9572**
- Training Accuracy: **0.9581**
- Final Overfitting Gap: **0.0009**

### Classification Report (6-class)

          precision    recall  f1-score   support

       0       1.00      1.00      1.00      4860
       1       0.97      0.98      0.97      9879
       2       0.96      0.93      0.94      8276
       3       0.96      0.96      0.96      4230
       4       0.88      0.92      0.90      4073
       5       1.00      0.97      0.99        80

accuracy                           0.96     31398
macro avg      0.96      0.96      0.96     31398
weighted avg   0.96      0.96      0.96     31398


---

## 📊 Target distribution (Binary)

0.0 24301
1.0 132685
Name: Attack_label, dtype: int64


- Class imbalance ratio: **0.183**
- ⚠️ Moderate class imbalance detected - monitoring recommended

---

# 🚀 Training LightGBM Classifier (Binary)

### 🔍 Enhanced Cross-Validation
- Cross-validation Accuracy: **0.9993 (+/- 0.0001)**
- Cross-validation Precision: **0.9991 (+/- 0.0001)**
- Cross-validation Recall: **1.0000 (+/- 0.0000)**
- Cross-validation F1-Score: **0.9996 (+/- 0.0000)**

**Overfitting Analysis**
- Training Score: **0.9993**
- Validation Score: **0.9993**
- Overfitting Gap: **0.0000**
- CV Variance: **0.000000**
- ✅ No significant overfitting detected

### 📊 FINAL RESULTS
- Training completed in **11.36 seconds**
- Prediction completed in **0.02 seconds**
- Test Accuracy: **0.9993**
- Training Accuracy: **0.9993**
- Final Overfitting Gap: **-0.0000**

### Classification Report (Binary)

          precision    recall  f1-score   support

     0.0       1.00      1.00      1.00      4860
     1.0       1.00      1.00      1.00     26538

accuracy                           1.00     31398
macro avg      1.00      1.00      1.00     31398
weighted avg   1.00      1.00      1.00     31398



---

# 📋 Summary Table

| Model     | Test_Acc(%) | F1_Score(%) | Std_Acc(%) | Std_F1(%) | Flash(KB) | RAM(MB) | OPS(K) |
|-----------|-------------|-------------|------------|-----------|-----------|---------|--------|
| 15-class  | 95.25       | 94.74       | 0.13       | 0.00      | 74.93     | 1.13    | 1.20   |
| 6-class   | 95.72       | 96.05       | 0.10       | 0.00      | 30.02     | 1.13    | 0.48   |
| binary    | 99.93       | 99.96       | 0.00       | 0.00      | 5.02      | 1.13    | 0.08   |

---
