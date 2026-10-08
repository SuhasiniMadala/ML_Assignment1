import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

print("=== STAGE 2: SYSTEMATIC OLS SWEEP ACROSS DEGREES (model2_ols_sweep.py) ===")

train1 = pd.read_excel('Train1.xlsx')
X1 = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
y1 = train1['y'].values

train2 = pd.read_excel('Train2.xlsx')
X2 = train2[['x1', 'x2', 'x3']].values
y2 = train2['y'].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- var1 (all 6 features) ---")
for d in range(1, 7):
    poly = PolynomialFeatures(degree=d)
    X1_p = poly.fit_transform(X1)
    val_mse, val_r2 = [], []
    for tr, val in kf.split(X1_p):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X1_p[tr])
        X_val = sc.transform(X1_p[val])
        m = LinearRegression()
        m.fit(X_tr, y1[tr])
        p_val = m.predict(X_val)
        val_mse.append(mean_squared_error(y1[val], p_val))
        val_r2.append(r2_score(y1[val], p_val))
    print(f"Deg {d} ({X1_p.shape[1]} terms): 5-Fold CV R² = {np.mean(val_r2):.4f}, CV MSE = {np.mean(val_mse):.4f}")

print("\n--- var2 (all 3 coordinates) ---")
for d in range(1, 10):
    poly = PolynomialFeatures(degree=d)
    X2_p = poly.fit_transform(X2)
    val_mse, val_r2 = [], []
    for tr, val in kf.split(X2_p):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X2_p[tr])
        X_val = sc.transform(X2_p[val])
        m = LinearRegression()
        m.fit(X_tr, y2[tr])
        p_val = m.predict(X_val)
        val_mse.append(mean_squared_error(y2[val], p_val))
        val_r2.append(r2_score(y2[val], p_val))
    print(f"Deg {d} ({X2_p.shape[1]} terms): 5-Fold CV R² = {np.mean(val_r2):.4f}, CV MSE = {np.mean(val_mse):.4f}")
