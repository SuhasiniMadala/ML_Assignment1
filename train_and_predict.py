import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

print("=== STARTING POLYNOMIAL REGRESSION PIPELINE ===")

# 1. Load Datasets
train1 = pd.read_excel('Train1.xlsx')
test1 = pd.read_excel('Test1.xlsx')
train2 = pd.read_excel('Train2.xlsx')
test2 = pd.read_excel('Test2.xlsx')

X1_train = train1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
y1_train = train1['y'].values
X1_test = test1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values

X2_train = train2[['x1', 'x2', 'x3']].values
y2_train = train2['y'].values
X2_test = test2[['x1', 'x2', 'x3']].values

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# ---------------------------------------------------------
# PHASE 1: VAR1 PIPELINE
# ---------------------------------------------------------
print("\n--- Evaluating Phase 1 (var1) ---")
p1_degrees = range(1, 8)
p1_metrics = []

for deg in p1_degrees:
    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(X1_train)
    
    val_mse_ols, val_r2_ols, tr_mse_ols, tr_r2_ols = [], [], [], []
    val_mse_ridge, val_r2_ridge, tr_mse_ridge, tr_r2_ridge = [], [], [], []
    
    for tr_idx, val_idx in kf.split(X1_train):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_poly[tr_idx])
        X_val = scaler.transform(X_poly[val_idx])
        
        # OLS (if feasible)
        if X_poly.shape[1] < 900:
            ols = LinearRegression()
            ols.fit(X_tr, y1_train[tr_idx])
            p_tr = ols.predict(X_tr)
            p_val = ols.predict(X_val)
            tr_mse_ols.append(mean_squared_error(y1_train[tr_idx], p_tr))
            tr_r2_ols.append(r2_score(y1_train[tr_idx], p_tr))
            val_mse_ols.append(mean_squared_error(y1_train[val_idx], p_val))
            val_r2_ols.append(r2_score(y1_train[val_idx], p_val))
        else:
            tr_mse_ols.append(np.nan)
            tr_r2_ols.append(np.nan)
            val_mse_ols.append(np.nan)
            val_r2_ols.append(np.nan)
            
        # Ridge (alpha=12.0)
        ridge = Ridge(alpha=12.0)
        ridge.fit(X_tr, y1_train[tr_idx])
        p_tr_r = ridge.predict(X_tr)
        p_val_r = ridge.predict(X_val)
        tr_mse_ridge.append(mean_squared_error(y1_train[tr_idx], p_tr_r))
        tr_r2_ridge.append(r2_score(y1_train[tr_idx], p_tr_r))
        val_mse_ridge.append(mean_squared_error(y1_train[val_idx], p_val_r))
        val_r2_ridge.append(r2_score(y1_train[val_idx], p_val_r))
        
    p1_metrics.append({
        'degree': deg,
        'features': X_poly.shape[1],
        'ols_tr_mse': np.nanmean(tr_mse_ols),
        'ols_tr_r2': np.nanmean(tr_r2_ols),
        'ols_val_mse': np.nanmean(val_mse_ols),
        'ols_val_r2': np.nanmean(val_r2_ols),
        'ridge_tr_mse': np.mean(tr_mse_ridge),
        'ridge_tr_r2': np.mean(tr_r2_ridge),
        'ridge_val_mse': np.mean(val_mse_ridge),
        'ridge_val_r2': np.mean(val_r2_ridge),
    })

p1_df = pd.DataFrame(p1_metrics)
print(p1_df.to_string())

# Train Best Model for var1: Degree 5 with Ridge(alpha=12.0)
best_deg1 = 5
poly1 = PolynomialFeatures(degree=best_deg1)
scaler1 = StandardScaler()

X1_tr_poly = scaler1.fit_transform(poly1.fit_transform(X1_train))
X1_te_poly = scaler1.transform(poly1.transform(X1_test))

model1 = Ridge(alpha=12.0)
model1.fit(X1_tr_poly, y1_train)
y1_pred = model1.predict(X1_te_poly)
y1_train_pred = model1.predict(X1_tr_poly)

print(f"\nPhase 1 Final Fit: Train MSE = {mean_squared_error(y1_train, y1_train_pred):.4f}, R2 = {r2_score(y1_train, y1_train_pred):.4f}")

# ---------------------------------------------------------
# PHASE 2: VAR2 PIPELINE
# ---------------------------------------------------------
print("\n--- Evaluating Phase 2 (var2) ---")
p2_degrees = range(1, 16)
p2_metrics = []

for deg in p2_degrees:
    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(X2_train)
    
    val_mse_ols, val_r2_ols, tr_mse_ols, tr_r2_ols = [], [], [], []
    val_mse_ridge, val_r2_ridge, tr_mse_ridge, tr_r2_ridge = [], [], [], []
    
    for tr_idx, val_idx in kf.split(X2_train):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_poly[tr_idx])
        X_val = scaler.transform(X_poly[val_idx])
        
        # OLS (if feasible)
        if X_poly.shape[1] < 900:
            ols = LinearRegression()
            ols.fit(X_tr, y2_train[tr_idx])
            p_tr = ols.predict(X_tr)
            p_val = ols.predict(X_val)
            tr_mse_ols.append(mean_squared_error(y2_train[tr_idx], p_tr))
            tr_r2_ols.append(r2_score(y2_train[tr_idx], p_tr))
            val_mse_ols.append(mean_squared_error(y2_train[val_idx], p_val))
            val_r2_ols.append(r2_score(y2_train[val_idx], p_val))
        else:
            tr_mse_ols.append(np.nan)
            tr_r2_ols.append(np.nan)
            val_mse_ols.append(np.nan)
            val_r2_ols.append(np.nan)
            
        # Ridge (alpha=1.0)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, y2_train[tr_idx])
        p_tr_r = ridge.predict(X_tr)
        p_val_r = ridge.predict(X_val)
        tr_mse_ridge.append(mean_squared_error(y2_train[tr_idx], p_tr_r))
        tr_r2_ridge.append(r2_score(y2_train[tr_idx], p_tr_r))
        val_mse_ridge.append(mean_squared_error(y2_train[val_idx], p_val_r))
        val_r2_ridge.append(r2_score(y2_train[val_idx], p_val_r))
        
    p2_metrics.append({
        'degree': deg,
        'features': X_poly.shape[1],
        'ols_tr_mse': np.nanmean(tr_mse_ols),
        'ols_tr_r2': np.nanmean(tr_r2_ols),
        'ols_val_mse': np.nanmean(val_mse_ols),
        'ols_val_r2': np.nanmean(val_r2_ols),
        'ridge_tr_mse': np.mean(tr_mse_ridge),
        'ridge_tr_r2': np.mean(tr_r2_ridge),
        'ridge_val_mse': np.mean(val_mse_ridge),
        'ridge_val_r2': np.mean(val_r2_ridge),
    })

p2_df = pd.DataFrame(p2_metrics)
print(p2_df.to_string())

# Train Best Model for var2: Degree 11 with Ridge(alpha=1.0)
best_deg2 = 11
poly2 = PolynomialFeatures(degree=best_deg2)
scaler2 = StandardScaler()

X2_tr_poly = scaler2.fit_transform(poly2.fit_transform(X2_train))
X2_te_poly = scaler2.transform(poly2.transform(X2_test))

model2 = Ridge(alpha=1.0)
model2.fit(X2_tr_poly, y2_train)
y2_pred = model2.predict(X2_te_poly)
y2_train_pred = model2.predict(X2_tr_poly)

print(f"\nPhase 2 Final Fit: Train MSE = {mean_squared_error(y2_train, y2_train_pred):.4f}, R2 = {r2_score(y2_train, y2_train_pred):.4f}")

# ---------------------------------------------------------
# SAVE PREDICTIONS
# ---------------------------------------------------------
sub1 = pd.DataFrame({'y': y1_pred})
sub2 = pd.DataFrame({'y': y2_pred})

sub1.to_csv('pred_var1.csv', index=False)
sub2.to_csv('pred_var2.csv', index=False)

sub1.to_csv('ROLLNO_pred_var1.csv', index=False)
sub2.to_csv('ROLLNO_pred_var2.csv', index=False)

print("\nSaved prediction CSVs:")
print("- pred_var1.csv (shape:", sub1.shape, ")")
print("- pred_var2.csv (shape:", sub2.shape, ")")

# Save metric summary tables to CSV for report generation
p1_df.to_csv('p1_metrics.csv', index=False)
p2_df.to_csv('p2_metrics.csv', index=False)

# ---------------------------------------------------------
# GENERATE PLOTS FOR REPORT
# ---------------------------------------------------------
print("\n--- Generating Plots ---")

# Figure 1: Var1 Degree vs MSE & R2
fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
color = '#1f77b4'
ax1.set_xlabel('Polynomial Degree', fontsize=11, fontweight='bold')
ax1.set_ylabel('Validation MSE', color=color, fontsize=11, fontweight='bold')
line1 = ax1.plot(p1_df['degree'], p1_df['ridge_val_mse'], color=color, marker='o', linewidth=2, label='Ridge Val MSE')
line2 = ax1.plot(p1_df['degree'], p1_df['ols_val_mse'], color='#ff7f0e', linestyle='--', marker='s', linewidth=1.5, label='OLS Val MSE')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 5)

ax2 = ax1.twinx()
color = '#2ca02c'
ax2.set_ylabel('Validation R² Score', color=color, fontsize=11, fontweight='bold')
line3 = ax2.plot(p1_df['degree'], p1_df['ridge_val_r2'], color=color, marker='^', linewidth=2, label='Ridge Val R²')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1.05)

# Highlight degree 5
ax1.axvline(x=5, color='#d62728', linestyle=':', linewidth=2, label='Selected Optimal (Degree 5)')

lines = line1 + line2 + line3 + [ax1.get_lines()[-1]]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.9)

plt.title('Phase 1 (var1): Model Validation Performance vs. Degree', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('p1_degree_vs_error.png')
plt.close()

# Figure 2: Var2 Degree vs MSE & R2
fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
color = '#1f77b4'
ax1.set_xlabel('Polynomial Degree', fontsize=11, fontweight='bold')
ax1.set_ylabel('Validation MSE', color=color, fontsize=11, fontweight='bold')
line1 = ax1.plot(p2_df['degree'], p2_df['ridge_val_mse'], color=color, marker='o', linewidth=2, label='Ridge Val MSE')
line2 = ax1.plot(p2_df['degree'][:10], p2_df['ols_val_mse'][:10], color='#ff7f0e', linestyle='--', marker='s', linewidth=1.5, label='OLS Val MSE')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 5)

ax2 = ax1.twinx()
color = '#2ca02c'
ax2.set_ylabel('Validation R² Score', color=color, fontsize=11, fontweight='bold')
line3 = ax2.plot(p2_df['degree'], p2_df['ridge_val_r2'], color=color, marker='^', linewidth=2, label='Ridge Val R²')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1.05)

# Highlight degree 11
ax1.axvline(x=11, color='#d62728', linestyle=':', linewidth=2, label='Selected Optimal (Degree 11)')

lines = line1 + line2 + line3 + [ax1.get_lines()[-1]]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.9)

plt.title('Phase 2 (var2): Model Validation Performance vs. Degree', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('p2_degree_vs_error.png')
plt.close()

# Figure 3: Actual vs Predicted Residual Plot for Phase 1
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
ax.scatter(y1_train, y1_train_pred, alpha=0.5, color='#1f77b4', edgecolors='none', s=25)
ax.plot([y1_train.min(), y1_train.max()], [y1_train.min(), y1_train.max()], 'r--', lw=2, label='Ideal Fit (y = ŷ)')
ax.set_xlabel('Actual Net Power Score (y)', fontsize=11, fontweight='bold')
ax.set_ylabel('Predicted Net Power Score (ŷ)', fontsize=11, fontweight='bold')
ax.set_title('Phase 1: Actual vs. Fitted Values (Degree 5 Ridge)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('p1_pred_vs_actual.png')
plt.close()

# Figure 4: Actual vs Predicted Residual Plot for Phase 2
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
ax.scatter(y2_train, y2_train_pred, alpha=0.5, color='#2ca02c', edgecolors='none', s=25)
ax.plot([y2_train.min(), y2_train.max()], [y2_train.min(), y2_train.max()], 'r--', lw=2, label='Ideal Fit (y = ŷ)')
ax.set_xlabel('Actual Thermal Anomaly Score (y)', fontsize=11, fontweight='bold')
ax.set_ylabel('Predicted Thermal Anomaly Score (ŷ)', fontsize=11, fontweight='bold')
ax.set_title('Phase 2: Actual vs. Fitted Values (Degree 11 Ridge)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('p2_pred_vs_actual.png')
plt.close()

# Figure 5: 3D Geological Surface Scatter Plot for Phase 2
fig = plt.figure(figsize=(7, 5), dpi=300)
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X2_train[:, 0], X2_train[:, 1], X2_train[:, 2], c=y2_train, cmap='viridis', s=20, alpha=0.8)
cb = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
cb.set_label('Thermal Anomaly Score (y)', fontsize=10, fontweight='bold')
ax.set_xlabel('X1 (East-West offset)', fontsize=9)
ax.set_ylabel('X2 (North-South offset)', fontsize=9)
ax.set_zlabel('X3 (Vertical depth offset)', fontsize=9)
ax.set_title('3D Subterranean Geothermal Anomaly Map (Phase 2)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('p2_3d_spatial_map.png')
plt.close()

print("All plots and models generated successfully!")
