import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

print("=== STAGE 1: BASELINE USING PROBLEM HINTS (model1_baseline.py) ===")

train1 = pd.read_excel('Train1.xlsx')
X1 = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
y1 = train1['y'].values

train2 = pd.read_excel('Train2.xlsx')
X2 = train2[['x1', 'x2', 'x3']].values
y2 = train2['y'].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# var1 Baseline: Degree 3 on x1-x3 (cols 0, 1, 2), dropping x4, x5, x6
poly1 = PolynomialFeatures(degree=3)
X1_b = poly1.fit_transform(X1[:, :3])

val_mse1, val_r21, tr_mse1, tr_r21 = [], [], [], []
for tr, val in kf.split(X1_b):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X1_b[tr])
    X_val = sc.transform(X1_b[val])
    model = LinearRegression()
    model.fit(X_tr, y1[tr])
    p_tr, p_val = model.predict(X_tr), model.predict(X_val)
    tr_mse1.append(mean_squared_error(y1[tr], p_tr))
    tr_r21.append(r2_score(y1[tr], p_tr))
    val_mse1.append(mean_squared_error(y1[val], p_val))
    val_r21.append(r2_score(y1[val], p_val))

print(f"var1 Baseline (Degree 3 on x1-x3, 20 terms):")
print(f"  Train R² = {np.mean(tr_r21):.4f}, Train MSE = {np.mean(tr_mse1):.4f}")
print(f"  5-Fold CV R² = {np.mean(val_r21):.4f} ± {np.std(val_r21):.4f}, CV MSE = {np.mean(val_mse1):.4f} ± {np.std(val_mse1):.4f}")

# var2 Baseline: Degree 4 on x1 only (col 0), dropping x2, x3
poly2 = PolynomialFeatures(degree=4)
X2_b = poly2.fit_transform(X2[:, :1])

val_mse2, val_r22, tr_mse2, tr_r22 = [], [], [], []
for tr, val in kf.split(X2_b):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X2_b[tr])
    X_val = sc.transform(X2_b[val])
    model = LinearRegression()
    model.fit(X_tr, y2[tr])
    p_tr, p_val = model.predict(X_tr), model.predict(X_val)
    tr_mse2.append(mean_squared_error(y2[tr], p_tr))
    tr_r22.append(r2_score(y2[tr], p_tr))
    val_mse2.append(mean_squared_error(y2[val], p_val))
    val_r22.append(r2_score(y2[val], p_val))

print(f"\nvar2 Baseline (Degree 4 on x1 only, 5 terms):")
print(f"  Train R² = {np.mean(tr_r22):.4f}, Train MSE = {np.mean(tr_mse2):.4f}")
print(f"  5-Fold CV R² = {np.mean(val_r22):.4f} ± {np.std(val_r22):.4f}, CV MSE = {np.mean(val_mse2):.4f} ± {np.std(val_mse2):.4f}")
