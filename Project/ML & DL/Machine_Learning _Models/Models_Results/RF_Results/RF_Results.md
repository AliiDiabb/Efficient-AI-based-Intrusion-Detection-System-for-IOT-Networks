# Training Random Forest Classifier

---

## 📂 Dataset Info
- **Total samples:** 156986  
- **Total features:** 49  
- **Missing values:** 0  
- **Target (15-class):** `Attack_type`  
- **Features shape:** (156986, 47)  

**Data split:**
- Training set: 125588 samples  
- Testing set: 31398 samples  

---

## 🎯 Target Distribution (15-class)

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



- **Class imbalance ratio:** 0.016 ⚠️ Severe imbalance detected  

---

## 🔍 Enhanced Cross-Validation (15-class)

- Accuracy: **0.9398** (+/- 0.0029)  
- Precision: **0.9393** (+/- 0.0039)  
- Recall: **0.9269** (+/- 0.0046)  
- F1-Score: **0.9321** (+/- 0.0040)  

**Overfitting Analysis:**
- Training Score: 0.9501  
- Validation Score: 0.9398  
- Overfitting Gap: 0.0103  
- CV Variance: 0.000002  
✅ No significant overfitting detected  

- Training time: 51.14s  
- Prediction time: 0.04s  

**Final Scores:**
- Test Accuracy: 0.9412  
- Training Accuracy: 0.9512  
- Out-of-Bag Score: 0.5874  
- Final Overfitting Gap: 0.0100  

---

## 📊 Classification Report (15-class)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00 | 0.96 | 0.98 | 2039 |
| 1     | 0.83 | 0.85 | 0.84 | 2112 |
| 2     | 1.00 | 1.00 | 1.00 | 2818 |
| 3     | 1.00 | 1.00 | 1.00 | 2050 |
| 4     | 1.00 | 1.00 | 1.00 | 2900 |
| 5     | 0.98 | 0.79 | 0.88 | 200 |
| 6     | 1.00 | 1.00 | 1.00 | 80 |
| 7     | 1.00 | 1.00 | 1.00 | 4860 |
| 8     | 0.90 | 0.79 | 0.84 | 1998 |
| 9     | 0.92 | 0.96 | 0.94 | 2014 |
| 10    | 0.89 | 0.95 | 0.92 | 2185 |
| 11    | 0.86 | 0.89 | 0.87 | 2062 |
| 12    | 0.90 | 0.87 | 0.89 | 2054 |
| 13    | 0.99 | 0.97 | 0.98 | 2015 |
| 14    | 0.84 | 0.89 | 0.87 | 2011 |

**Accuracy:** 0.94 (31398 samples)  
**Macro avg:** Precision 0.94 | Recall 0.93 | F1 0.93  
**Weighted avg:** Precision 0.94 | Recall 0.94 | F1 0.94  

---

## 🎯 Target Distribution (6-class)

0 24301
1 49396
2 41378
3 21148
4 20363
5 400




- **Class imbalance ratio:** 0.008 ⚠️ Severe imbalance detected  

---

## 🔍 Enhanced Cross-Validation (6-class)

- Accuracy: **0.9443** (+/- 0.0029)  
- Precision: **0.9506** (+/- 0.0035)  
- Recall: **0.9491** (+/- 0.0050)  
- F1-Score: **0.9498** (+/- 0.0037)  

**Overfitting Analysis:**
- Training Score: 0.9546  
- Validation Score: 0.9443  
- Overfitting Gap: 0.0103  
- CV Variance: 0.000002  
✅ No significant overfitting detected  

- Training time: 39.09s  
- Prediction time: 0.02s  

**Final Scores:**
- Test Accuracy: 0.9453  
- Training Accuracy: 0.9544  
- Out-of-Bag Score: 0.6262  
- Final Overfitting Gap: 0.0090  

---

## 📊 Classification Report (6-class)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00 | 0.99 | 1.00 | 4860 |
| 1     | 0.96 | 0.96 | 0.96 | 9879 |
| 2     | 0.93 | 0.92 | 0.92 | 8276 |
| 3     | 0.97 | 0.95 | 0.96 | 4230 |
| 4     | 0.85 | 0.89 | 0.87 | 4073 |
| 5     | 1.00 | 0.97 | 0.99 | 80 |

**Accuracy:** 0.95 (31398 samples)  
**Macro avg:** Precision 0.95 | Recall 0.95 | F1 0.95  
**Weighted avg:** Precision 0.95 | Recall 0.95 | F1 0.95  

---

## 🎯 Target Distribution (Binary)

0.0 24301
1.0 132685




---

## 🔍 Enhanced Cross-Validation (Binary)

- Accuracy: **0.9991** (+/- 0.0002)  
- Precision: **0.9991** (+/- 0.0001)  
- Recall: **0.9998** (+/- 0.0002)  
- F1-Score: **0.9994** (+/- 0.0001)  

**Overfitting Analysis:**
- Training Score: 0.9992  
- Validation Score: 0.9991  
- Overfitting Gap: 0.0001  
- CV Variance: 0.000000  
✅ No significant overfitting detected  

- Training time: 31.83s  
- Prediction time: 0.02s  

**Final Scores:**
- Test Accuracy: 0.9992  
- Training Accuracy: 0.9992  
- Out-of-Bag Score: 0.6599  
- Final Overfitting Gap: -0.0000  

---

## 📊 Classification Report (Binary)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 1.00 | 1.00 | 1.00 | 4860 |
| 1.0   | 1.00 | 1.00 | 1.00 | 26538 |

**Accuracy:** 1.00 (31398 samples)  
**Macro avg:** Precision 1.00 | Recall 1.00 | F1 1.00  
**Weighted avg:** Precision 1.00 | Recall 1.00 | F1 1.00  

---

## 📋 Summary Table

| Model     | Test_Acc(%) | F1_Score(%) | Std_Acc(%) | Std_F1(%) | Flash(KB) | RAM(KB) | OPS(K) |
|-----------|-------------|-------------|------------|-----------|-----------|---------|--------|
| 15-class  | 94.12       | 93.21       | 0.14       | 0.00      | 221.22    | 4.61    | 3.380  |
| 6-class   | 94.53       | 94.98       | 0.15       | 0.02      | 189.96    | 4.61    | 3.034  |
| Binary    | 99.92       | 99.94       | 0.01       | 0.00      | 6.25      | 1.84    | 0.1    |
