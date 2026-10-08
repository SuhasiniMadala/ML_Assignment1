import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

print("=== STAGE 3: REGULARIZED POLYNOMIAL MODELS (model3_regularized.py) ===")

train1 = pd.read_excel('Train1.xlsx')
X1 = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
y1 = train1['y'].values

train2 = pd.read_excel('Train2.xlsx')
X2 = train2[['x1', 'x2', 'x3']].values
y2 = train2['y'].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# var1: Degree 5 with Lasso (alpha=0.00348)
poly1 = PolynomialFeatures(degree=5)
X1_p = poly1.fit_transform(X1)

val_mse1, val_r21, active_terms1 = [], [], []
for tr, val in kf.split(X1_p):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X1_p[tr])
    X_val = sc.transform(X1_p[val])
    model = Lasso(alpha=0.00348, max_iter=20000, random_state=42)
    model.fit(X_tr, y1[tr])
    p_val = model.predict(X_val)
    val_mse1.append(mean_squared_error(y1[val], p_val))
    val_r21.append(r2_score(y1[val], p_val))
    active_terms1.append(np.sum(model.coef_ != 0))

print(f"var1 (Degree 5 Lasso, alpha=0.00348):")
print(f"  Active terms = {np.mean(active_terms1):.0f} / 462 ({100*(1-np.mean(active_terms1)/462):.1f}% pruned to zero)")
print(f"  5-Fold CV R² = {np.mean(val_r21):.4f} ± {np.std(val_r21):.4f}, CV MSE = {np.mean(val_mse1):.4f} ± {np.std(val_mse1):.4f}")

# var2: Degree 9 with Ridge (alpha=0.00665)
poly2 = PolynomialFeatures(degree=9)
X2_p = poly2.fit_transform(X2)

val_mse2, val_r22, weight_norms2 = [], [], []
for tr, val in kf.split(X2_p):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X2_p[tr])
    X_val = sc.transform(X2_p[val])
    model = Ridge(alpha=0.00665, random_state=42)
    model.fit(X_tr, y2[tr])
    p_val = model.predict(X_val)
    val_mse2.append(mean_squared_error(y2[val], p_val))
    val_r22.append(r2_score(y2[val], p_val))
    weight_norms2.append(np.linalg.norm(model.coef_))

print(f"\nvar2 (Degree 9 Ridge, alpha=0.00665):")
print(f"  Weight norm ||w||_2 = {np.mean(weight_norms2):.2f}")
print(f"  5-Fold CV R² = {np.mean(val_r22):.4f} ± {np.std(val_r22):.4f}, CV MSE = {np.mean(val_mse2):.4f} ± {np.std(val_mse2):.4f}")
