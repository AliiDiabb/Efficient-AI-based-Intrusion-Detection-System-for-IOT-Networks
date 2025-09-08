# 🎯 XGBoost Classification Experiments on Edge-IIoTset

---

## Dataset Info
- **Total samples:** 156986  
- **Total features:** 51  
- **Missing values:** 0  
- Using **Attack_type** for **15-class classification**  
- **Features shape:** (156986, 47)  

---

## 📊 Target Distribution (15-class)
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

## 🔬 Training XGBoost Classifier (15-class)

**Cross-Validation Results**
- Accuracy: **0.9511 ± 0.0023**  
- Precision: **0.9468 ± 0.0022**  
- Recall: **0.9391 ± 0.0044**  
- F1-Score: **0.9442 ± 0.0032**

**Overfitting Analysis**
- Overfitting Gap: **0.0011**  
- CV Variance: **0.000001**  
- ✅ No significant overfitting detected  
- Final Overfitting Gap: **0.0019**

**Classification Report**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 0.96   | 0.98     | 2039    |
| 1     | 0.82      | 0.87   | 0.85     | 2112    |
| 2     | 1.00      | 0.99   | 1.00     | 2818    |
| 3     | 1.00      | 1.00   | 1.00     | 2050    |
| 4     | 1.00      | 1.00   | 1.00     | 2900    |
| 5     | 0.91      | 0.82   | 0.87     | 200     |
| 6     | 1.00      | 1.00   | 1.00     | 80      |
| 7     | 1.00      | 1.00   | 1.00     | 4860    |
| 8     | 0.91      | 0.82   | 0.86     | 1998    |
| 9     | 0.92      | 0.96   | 0.94     | 2014    |
| 10    | 0.89      | 0.95   | 0.92     | 2185    |
| 11    | 0.90      | 0.88   | 0.89     | 2062    |
| 12    | 0.95      | 0.90   | 0.93     | 2054    |
| 13    | 0.99      | 0.96   | 0.98     | 2015    |
| 14    | 0.86      | 0.94   | 0.90     | 2011    |

**Accuracy:** 0.95  

---

## 📊 Target Distribution (6-class)
0 24301
1 49396
2 41378
3 21148
4 20363
5 400

- **Class imbalance ratio:** 0.008 ⚠️ Severe imbalance detected  

---

## 🔬 Training XGBoost Classifier (6-class)

**Cross-Validation Results**
- Accuracy: **0.9480 ± 0.0023**  
- Precision: **0.9555 ± 0.0017**  
- Recall: **0.9530 ± 0.0050**  
- F1-Score: **0.9542 ± 0.0032**

**Overfitting Analysis**
- Overfitting Gap: **0.0008**  
- CV Variance: **0.000001**  
- ✅ No significant overfitting detected  
- Final Overfitting Gap: **-0.0002**

**Classification Report**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 0.99   | 1.00     | 4860    |
| 1     | 0.96      | 0.96   | 0.96     | 9879    |
| 2     | 0.93      | 0.93   | 0.93     | 8276    |
| 3     | 0.96      | 0.96   | 0.96     | 4230    |
| 4     | 0.88      | 0.90   | 0.89     | 4073    |
| 5     | 1.00      | 0.97   | 0.99     | 80      |

**Accuracy:** 0.95  

---

## 📊 Target Distribution (Binary)
0.0 24301
1.0 132685


- **Class imbalance ratio:** 0.183 ⚠️ Moderate imbalance detected  

---

## 🔬 Training XGBoost Classifier (Binary)

**Cross-Validation Results**
- Accuracy: **0.9991 ± 0.0003**  
- Precision: **0.9989 ± 0.0004**  
- Recall: **1.0000 ± 0.0000**  
- F1-Score: **0.9995 ± 0.0002**

**Overfitting Analysis**
- Overfitting Gap: **0.0000**  
- CV Variance: **0.000000**  
- ✅ No significant overfitting detected  
- Final Overfitting Gap: **0.0001**

**Classification Report**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 1.00      | 0.99   | 1.00     | 4860    |
| 1.0   | 1.00      | 1.00   | 1.00     | 26538   |

**Accuracy:** 1.00  

---

## 📋 SUMMARY TABLE

| Model    | Test_Acc(%) | F1_Score(%) | Std_Acc(%) | Std_F1(%) | Flash(KB) | RAM(KB) | OPS(K) |
|----------|-------------|-------------|------------|-----------|-----------|---------|--------|
| 15-class | 95.11       | 94.42       | 0.12       | 0.00      | 266.59    | 0.51    | 4.27   |
| 6-class  | 95.48       | 95.94       | 0.13       | 0.00      | 110.35    | 0.51    | 1.77   |
| binary   | 99.91       | 99.95       | 0.01       | 0.01      | 18.22     | 0.51    | 0.29   |
