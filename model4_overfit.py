import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

print("=== STAGE 4: TESTING OVERFITTING & UNCONSTRAINED OLS (model4_overfit.py) ===")

train1 = pd.read_excel('Train1.xlsx')
X1 = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
y1 = train1['y'].values

train2 = pd.read_excel('Train2.xlsx')
X2 = train2[['x1', 'x2', 'x3']].values
y2 = train2['y'].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# var1: Degree 8 OLS (3,003 terms)
poly1 = PolynomialFeatures(degree=8)
X1_p = poly1.fit_transform(X1)

val_mse1, val_r21, tr_mse1, tr_r21 = [], [], [], []
for tr, val in kf.split(X1_p):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X1_p[tr])
    X_val = sc.transform(X1_p[val])
    model = LinearRegression()
    model.fit(X_tr, y1[tr])
    p_tr, p_val = model.predict(X_tr), model.predict(X_val)
    tr_mse1.append(mean_squared_error(y1[tr], p_tr))
    tr_r21.append(r2_score(y1[tr], p_tr))
    val_mse1.append(mean_squared_error(y1[val], p_val))
    val_r21.append(r2_score(y1[val], p_val))

print(f"var1 Overfit Demo (Degree 8 OLS, 3,003 terms):")
print(f"  Train R² = {np.mean(tr_r21):.4f}, Train MSE = {np.mean(tr_mse1):.6f}")
print(f"  5-Fold CV R² = {np.mean(val_r21):.4f} ± {np.std(val_r21):.4f}, CV MSE = {np.mean(val_mse1):.4f} ± {np.std(val_mse1):.4f}")

# var2: Degree 15 OLS (816 terms)
poly2 = PolynomialFeatures(degree=15)
X2_p = poly2.fit_transform(X2)

val_mse2, val_r22, tr_mse2, tr_r22 = [], [], [], []
for tr, val in kf.split(X2_p):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X2_p[tr])
    X_val = sc.transform(X2_p[val])
    model = LinearRegression()
    model.fit(X_tr, y2[tr])
    p_tr, p_val = model.predict(X_tr), model.predict(X_val)
    tr_mse2.append(mean_squared_error(y2[tr], p_tr))
    tr_r22.append(r2_score(y2[tr], p_tr))
    val_mse2.append(mean_squared_error(y2[val], p_val))
    val_r22.append(r2_score(y2[val], p_val))

print(f"\nvar2 Overfit Demo (Degree 15 OLS, 816 terms):")
print(f"  Train R² = {np.mean(tr_r22):.4f}, Train MSE = {np.mean(tr_mse2):.6f}")
print(f"  5-Fold CV R² = {np.mean(val_r22):.4f} ± {np.std(val_r22):.4f}, CV MSE = {np.mean(val_mse2):.4f} ± {np.std(val_mse2):.4f}")
