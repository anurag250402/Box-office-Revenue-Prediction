# 🎬 Box Office Revenue Prediction Using Linear Regression

This project uses machine learning to predict the **box office revenue of movies** using a range of features like genre, distributor, release day, and more. The primary focus is on using **Linear Regression**, while also comparing it to **XGBoost** for better accuracy.

---

## 📌 Project Objective

- Predict the domestic box office revenue of movies using historical data.
- Identify which features have the most influence on box office performance.
- Compare Linear Regression with a powerful ensemble model (XGBoost).
  
---

## 🛠️ Libraries Used

- `pandas`, `numpy` – for data handling and manipulation  
- `matplotlib`, `seaborn` – for visualizations  
- `scikit-learn` – for preprocessing, model training, and evaluation  
- `xgboost` – for training a high-performance regression model  
- `warnings` – to suppress unnecessary warning messages  

---

## 📂 Dataset Overview

The dataset includes details such as:

- `title`: Movie title  
- `domestic_revenue`: Revenue in the US  
- `world_revenue`: Global revenue  
- `distributor`: Distribution company  
- `opening_revenue`: Opening weekend revenue  
- `opening_theaters`: Number of theaters  
- `budget`: Budget (removed due to missing data)  
- `MPAA`: Movie rating (G, PG, R, etc.)  
- `genres`: Movie genres  
- `release_days`: Days since release  

---

## 🧹 Data Preprocessing Steps

1. **Missing Data Handling**:
   - Dropped `budget` due to many missing values.
   - Filled missing `MPAA` and `genres` with mode.
   - Dropped any remaining null rows.

2. **Data Type Cleaning**:
   - Removed symbols like `$` and `,` from revenue columns.
   - Converted those columns to numeric format.

3. **Log Transformation**:
   - Applied `log10` to reduce skewness in revenue-related columns.

4. **Genre Feature Extraction**:
   - Used `CountVectorizer` to convert genres into individual binary columns.
   - Removed genre columns with >95% zero values to reduce noise.

5. **Label Encoding**:
   - Encoded categorical variables like `distributor` and `MPAA`.

6. **Feature Scaling**:
   - Used `StandardScaler` to normalize feature values.

---

## 📊 Exploratory Data Analysis (EDA)

- **Count Plots & Distributions**: To understand the spread and skew of numerical features.
- **Box Plots**: To check for outliers.
- **Correlation Matrix**: To detect multicollinearity among features.

---

## 🧠 Model Development

- **Train-Test Split**:
  - 90% Training, 10% Validation

```python
from sklearn.model_selection import train_test_split

features = df.drop(['title', 'domestic_revenue'], axis=1)
target = df['domestic_revenue'].values

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)

```

---

## 📌 Models Used

- Linear Regression  
- XGBoost Regressor  

---

## 📌 Feature Normalization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
```
---

## 📈 Model Evaluation
Metric Used: Mean Absolute Error (MAE)

```bash
Training Error   : 0.2104
Validation Error : 0.6358
These values are calculated using log-transformed revenues. Actual revenue errors will be larger due to the logarithmic scale.
```
---

## ✅ Key Takeaways

- Movies with **PG** and **R** ratings tend to earn higher revenues.
- **Log-transforming** skewed features and **scaling** helps improve model performance.
- **XGBoost** performed better on unseen data compared to basic regression.

---

## 📁 Files Included

- `Box_office_Revenue_Prediction_ML.ipynb` – Full code and walkthrough  
- `README.md` – This project overview  
- `boxoffice` – The movie dataset  

---

## 🚀 How to Run

### Clone the repository

```bash
git clone https://github.com/yourusername/box-office-revenue-prediction.git
```
