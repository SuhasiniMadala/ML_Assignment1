import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge

ROLLNO = "BT2024043"

print(f"=== WINNING MODEL TRAINING & TEST PREDICTION PIPELINE ({ROLLNO}) ===")

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

# ---------------------------------------------------------
# Phase 1: Degree 5 Lasso (alpha=0.00348)
# ---------------------------------------------------------
poly1 = PolynomialFeatures(degree=5)
scaler1 = StandardScaler()

X1_tr_p = scaler1.fit_transform(poly1.fit_transform(X1_train))
X1_te_p = scaler1.transform(poly1.transform(X1_test))

model1 = Lasso(alpha=0.00348, max_iter=20000, random_state=42)
model1.fit(X1_tr_p, y1_train)
y1_pred = model1.predict(X1_te_p)

# ---------------------------------------------------------
# Phase 2: Degree 9 Ridge (alpha=0.00665)
# ---------------------------------------------------------
poly2 = PolynomialFeatures(degree=9)
scaler2 = StandardScaler()

X2_tr_p = scaler2.fit_transform(poly2.fit_transform(X2_train))
X2_te_p = scaler2.transform(poly2.transform(X2_test))

model2 = Ridge(alpha=0.00665, random_state=42)
model2.fit(X2_tr_p, y2_train)
y2_pred = model2.predict(X2_te_p)

# Save Prediction Files
file1 = f"{ROLLNO}_pred_var1.csv"
file2 = f"{ROLLNO}_pred_var2.csv"

df1 = pd.DataFrame({'y': y1_pred})
df2 = pd.DataFrame({'y': y2_pred})

df1.to_csv(file1, index=False)
df2.to_csv(file2, index=False)

print("\n--- Final Prediction Files Generated ---")
print(f"1. {file1}: Shape = {df1.shape}, NaNs = {df1.isna().sum().sum()}, Mean = {df1['y'].mean():.2f}, Std = {df1['y'].std():.2f}, Range = [{df1['y'].min():.2f}, {df1['y'].max():.2f}]")
print(f"2. {file2}: Shape = {df2.shape}, NaNs = {df2.isna().sum().sum()}, Mean = {df2['y'].mean():.2f}, Std = {df2['y'].std():.2f}, Range = [{df2['y'].min():.2f}, {df2['y'].max():.2f}]")
