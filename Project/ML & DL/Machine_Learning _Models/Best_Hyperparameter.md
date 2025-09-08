# Best Hyperparameter Constraints for the Hardware

## Random Forest
| Parameter          | Value | Type      |
|--------------------|-------|-----------|
| min_samples_leaf   | 10    | float     |
| max_depth          | 25    | int       |
| max_features       | None  | NoneType  |
| n_estimators       | 2     | int       |
| min_samples_split  | 10    | int       |
| bootstrap          | True  | Boolean   |
| oob_score          | True  | Boolean   |
| class_weight       | None  | NoneType  |

---

## LightGBM
| Parameter          | Value | Type      |
|--------------------|-------|-----------|
| class_weight       | None  | NoneType  |
| colsample_bytree   | 0.6   | float     |
| learning_rate      | 0.05  | float     |
| max_depth          | -1    | int       |
| min_child_samples  | 40    | int       |
| n_estimators       | 100   | int       |
| reg_alpha          | 0     | int       |
| reg_lambda         | 0     | int       |
| subsample          | 0.7   | float     |

---

## XGBoost
| Parameter          | Value | Type      |
|--------------------|-------|-----------|
| colsample_bytree   | 0.7   | float     |
| gamma              | 0.1   | float     |
| learning_rate      | 0.7   | float     |
| max_depth          | 3     | int       |
| min_child_weight   | 3     | int       |
| n_estimators       | 23    | int       |
| reg_alpha          | 0     | int       |
| reg_lambda         | 0.01  | int       |
| subsample          | 0.8   | float     |
