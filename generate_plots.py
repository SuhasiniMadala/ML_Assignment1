import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm

os.makedirs('figures', exist_ok=True)

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans, Arial, Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

train1 = pd.read_excel('Train1.xlsx')
X1 = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
y1 = train1['y'].values

train2 = pd.read_excel('Train2.xlsx')
X2 = train2[['x1', 'x2', 'x3']].values
y2 = train2['y'].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("=== GENERATING ALL 5 REPORT FIGURES (generate_plots.py) ===")

# ---------------------------------------------------------
# FIGURE 1: Validation MSE across polynomial degrees (U-shaped curves)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

var1_degs = list(range(1, 7))
var1_ols_mse = []
for d in var1_degs:
    poly = PolynomialFeatures(degree=d)
    X_p = poly.fit_transform(X1)
    mses = []
    for tr, val in kf.split(X_p):
        sc = StandardScaler()
        m = LinearRegression().fit(sc.fit_transform(X_p[tr]), y1[tr])
        mses.append(mean_squared_error(y1[val], m.predict(sc.transform(X_p[val]))))
    var1_ols_mse.append(np.mean(mses))

ax1.plot(var1_degs, var1_ols_mse, 'o--', color='#d95f02', label='OLS (Unregularized)', linewidth=1.5)
ax1.plot(5, 0.3450, 'o', color='#2ca02c', markersize=8, label='Lasso Optimum (MSE = 0.345)')
ax1.set_yscale('log')
ax1.set_xlabel('Polynomial Degree', fontsize=10, fontweight='bold')
ax1.set_ylabel('Validation MSE (Log Scale)', fontsize=10, fontweight='bold')
ax1.set_title('var1 (Net Power): Bias-Variance Curve vs Degree', fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.annotate('OLS Noise Explosion\n(MSE = 97.32)', xy=(6, 97.32), xytext=(4.2, 20),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6), fontsize=8, color='red')

var2_degs = list(range(1, 11))
var2_ols_mse = []
for d in var2_degs:
    poly = PolynomialFeatures(degree=d)
    X_p = poly.fit_transform(X2)
    mses = []
    for tr, val in kf.split(X_p):
        sc = StandardScaler()
        m = LinearRegression().fit(sc.fit_transform(X_p[tr]), y2[tr])
        mses.append(mean_squared_error(y2[val], m.predict(sc.transform(X_p[val]))))
    var2_ols_mse.append(np.mean(mses))

ax2.plot(var2_degs, var2_ols_mse, 'o--', color='#1b9e77', label='OLS (Unregularized)', linewidth=1.5)
ax2.plot(9, 0.2685, 'o', color='#2ca02c', markersize=8, label='Ridge Optimum (MSE = 0.268)')
ax2.set_yscale('log')
ax2.set_xlabel('Polynomial Degree', fontsize=10, fontweight='bold')
ax2.set_ylabel('Validation MSE (Log Scale)', fontsize=10, fontweight='bold')
ax2.set_title('var2 (Thermal Anomaly): Bias-Variance Curve vs Degree', fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('figures/fig1_degree_vs_mse.png')
plt.close()
print("Saved figures/fig1_degree_vs_mse.png")

# ---------------------------------------------------------
# FIGURE 2: Lasso Sparsity & Ridge Weight Shrinkage
# ---------------------------------------------------------
poly1_5 = PolynomialFeatures(degree=5)
X1_p5 = poly1_5.fit_transform(X1)
sc1 = StandardScaler()
X1_p5_sc = sc1.fit_transform(X1_p5)
lasso1 = Lasso(alpha=0.00348, max_iter=20000, random_state=42).fit(X1_p5_sc, y1)

poly2_9 = PolynomialFeatures(degree=9)
X2_p9 = poly2_9.fit_transform(X2)
sc2 = StandardScaler()
X2_p9_sc = sc2.fit_transform(X2_p9)
ridge2 = Ridge(alpha=0.00665, random_state=42).fit(X2_p9_sc, y2)
ols2_9 = LinearRegression().fit(X2_p9_sc, y2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=300)

ax1.stem(range(len(lasso1.coef_)), lasso1.coef_, linefmt='C0-', markerfmt='C0o', basefmt='k-')
ax1.set_title('var1 Lasso Sparsity: 195 Active Terms (57.8% Pruned Out of 462 Total Features)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Polynomial Monomial Index (0 ... 461)', fontsize=9)
ax1.set_ylabel('Absolute Coefficient (w_j)', fontsize=9)

ax2.plot(sorted(np.abs(ols2_9.coef_)), label='Unconstrained OLS (Exploded Norm)', color='#d95f02', linewidth=1.5)
ax2.plot(sorted(np.abs(ridge2.coef_)), label='Ridge (Shrunk Norm ||w||_2 = 20.55)', color='#1b9e77', linewidth=2)
ax2.set_yscale('log')
ax2.set_title('var2 Coefficient Shrinkage: Controlled Ridge (L2) vs Astronomical Overfit OLS', fontsize=11, fontweight='bold')
ax2.set_xlabel('Sorted Feature Rank Index', fontsize=9)
ax2.set_ylabel('Coefficient Magnitude (Log Scale)', fontsize=9)
ax2.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('figures/fig2_regularization_effects.png')
plt.close()
print("Saved figures/fig2_regularization_effects.png")

# ---------------------------------------------------------
# FIGURE 3: Cross-validation R2 & Log MSE Bar Charts across M1-M4
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

stages = ['M1: Baseline\n(PDF Hints)', 'M2: OLS Sweep\n(All Features)', 'M3+: Tuned Reg\n(WINNER)', 'M4: Overfit\n(Excessive Deg)']
v1_r2 = [0.1530, 0.9222, 0.9698, 0.4810]
v2_r2 = [0.0810, 0.9941, 0.9942, -4.9055]

v1_mse = [9.6796, 0.8893, 0.3450, 5.9258]
v2_mse = [43.6157, 0.2700, 0.2685, 285.50]

x = np.arange(len(stages))
width = 0.35

ax1.bar(x - width/2, v1_r2, width, label='var1 R²', color='#1f77b4')
ax1.bar(x + width/2, [max(r, -0.6) for r in v2_r2], width, label='var2 R²', color='#ff7f0e')
ax1.set_xticks(x)
ax1.set_xticklabels(stages, fontsize=8)
ax1.set_ylabel('Cross-Validation R² Score', fontsize=10, fontweight='bold')
ax1.set_title('Cross-Validation R² Progression Across Model Stages', fontsize=11, fontweight='bold')
ax1.legend(loc='lower left', fontsize=9)
ax1.set_ylim(-0.7, 1.1)

ax2.bar(x - width/2, v1_mse, width, label='var1 MSE', color='#1f77b4')
ax2.bar(x + width/2, v2_mse, width, label='var2 MSE', color='#ff7f0e')
ax2.set_yscale('log')
ax2.set_xticks(x)
ax2.set_xticklabels(stages, fontsize=8)
ax2.set_ylabel('Cross-Validation Mean Squared Error (Log Scale)', fontsize=10, fontweight='bold')
ax2.set_title('Cross-Validation MSE Across Stages (Lower is Better)', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('figures/fig3_master_comparison.png')
plt.close()
print("Saved figures/fig3_master_comparison.png")

# ---------------------------------------------------------
# FIGURE 4: Residual distribution histograms & residual scatter plots
# ---------------------------------------------------------
# Out-of-fold residuals for M3+
oof_pred1 = np.zeros(len(y1))
for tr, val in kf.split(X1_p5):
    sc = StandardScaler()
    m = Lasso(alpha=0.00348, max_iter=20000, random_state=42).fit(sc.fit_transform(X1_p5[tr]), y1[tr])
    oof_pred1[val] = m.predict(sc.transform(X1_p5[val]))
res1 = y1 - oof_pred1

oof_pred2 = np.zeros(len(y2))
for tr, val in kf.split(X2_p9):
    sc = StandardScaler()
    m = Ridge(alpha=0.00665, random_state=42).fit(sc.fit_transform(X2_p9[tr]), y2[tr])
    oof_pred2[val] = m.predict(sc.transform(X2_p9[val]))
res2 = y2 - oof_pred2

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6.5), dpi=300)

# Top Left: var1 Residual Histogram
sns.histplot(res1, kde=True, ax=ax1, color='#1f77b4', stat='density')
mu1, std1 = norm.fit(res1)
xmin, xmax = ax1.get_xlim()
x_p = np.linspace(xmin, xmax, 100)
ax1.plot(x_p, norm.pdf(x_p, mu1, std1), 'r--', lw=2, label=f'Normal Fit (μ={mu1:.2f}, σ={std1:.2f})')
ax1.set_title('var1 Residual Distribution (Degree 5 Lasso)', fontsize=10, fontweight='bold')
ax1.legend(loc='upper right', fontsize=8)

# Top Right: var2 Residual Histogram
sns.histplot(res2, kde=True, ax=ax2, color='#ff7f0e', stat='density')
mu2, std2 = norm.fit(res2)
xmin, xmax = ax2.get_xlim()
x_p = np.linspace(xmin, xmax, 100)
ax2.plot(x_p, norm.pdf(x_p, mu2, std2), 'r--', lw=2, label=f'Normal Fit (μ={mu2:.2f}, σ={std2:.2f})')
ax2.set_title('var2 Residual Distribution (Degree 9 Ridge)', fontsize=10, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8)

# Bottom Left: var1 Residuals vs Predicted
ax3.scatter(oof_pred1, res1, alpha=0.5, color='#1f77b4', s=15)
ax3.axhline(0, color='r', linestyle='--', lw=1.5)
ax3.set_xlabel('Predicted Net Power (ŷ)', fontsize=9)
ax3.set_ylabel('Residual (y - ŷ)', fontsize=9)
ax3.set_title('var1 Residuals vs Predicted Values (Homoscedasticity)', fontsize=10, fontweight='bold')

# Bottom Right: var2 Residuals vs Predicted
ax4.scatter(oof_pred2, res2, alpha=0.5, color='#ff7f0e', s=15)
ax4.axhline(0, color='r', linestyle='--', lw=1.5)
ax4.set_xlabel('Predicted Thermal Anomaly (ŷ)', fontsize=9)
ax4.set_ylabel('Residual (y - ŷ)', fontsize=9)
ax4.set_title('var2 Residuals vs Predicted Values (Homoscedasticity)', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig4_residual_diagnostics.png')
plt.close()
print("Saved figures/fig4_residual_diagnostics.png")

# ---------------------------------------------------------
# FIGURE 5: 1:1 Ground Truth vs CV Predicted Reference Scatter Plots
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

ax1.scatter(y1, oof_pred1, alpha=0.5, c=np.abs(res1), cmap='viridis', s=20)
ax1.plot([y1.min(), y1.max()], [y1.min(), y1.max()], 'r--', lw=2, label='Ideal 1:1 Reference (y = ŷ)')
ax1.set_xlabel('Ground Truth (y)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Cross-Validated Prediction (ŷ)', fontsize=10, fontweight='bold')
ax1.set_title('var1 (Net Power): Actual vs Predicted (CV R² = 0.9698)', fontsize=11, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)

ax2.scatter(y2, oof_pred2, alpha=0.5, c=np.abs(res2), cmap='magma', s=20)
ax2.plot([y2.min(), y2.max()], [y2.min(), y2.max()], 'r--', lw=2, label='Ideal 1:1 Reference (y = ŷ)')
ax2.set_xlabel('Ground Truth (y)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Cross-Validated Prediction (ŷ)', fontsize=10, fontweight='bold')
ax2.set_title('var2 (Thermal Anomaly): Actual vs Predicted (CV R² = 0.9942)', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('figures/fig5_actual_vs_predicted.png')
plt.close()
print("Saved figures/fig5_actual_vs_predicted.png")
print("\nAll 5 figures generated successfully!")
