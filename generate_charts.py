import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Create images directory
if not os.path.exists('images'):
    os.makedirs('images')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10,6)

# Load data
df = pd.read_csv('data/train.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
df = df.drop_duplicates()
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

# Chart 1: Outlier Detection
plt.figure(figsize=(10,6))
sns.boxplot(x=df['Sales'])
plt.title('Outlier Detection in Sales', fontweight='bold')
plt.tight_layout()
plt.savefig('images/01_outlier_detection.png', dpi=100, bbox_inches='tight')
plt.close()

# Chart 2: Sales by Region
plt.figure(figsize=(10,6))
region_sales = df.groupby('Region')['Sales'].sum().sort_values()
sns.barplot(x=region_sales.index, y=region_sales.values)
plt.title('Total Sales by Region', fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/02_sales_by_region.png', dpi=100, bbox_inches='tight')
plt.close()

# Chart 3: Sales by Category
plt.figure(figsize=(10,6))
category_sales = df.groupby('Category')['Sales'].sum()
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title('Sales by Category', fontweight='bold')
plt.tight_layout()
plt.savefig('images/03_sales_by_category.png', dpi=100, bbox_inches='tight')
plt.close()

# Chart 4: Monthly Sales Trend
plt.figure(figsize=(10,6))
monthly_sales = df.groupby('Month')['Sales'].sum()
monthly_sales.plot()
plt.title('Monthly Sales Trend', fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.savefig('images/04_monthly_sales_trend.png', dpi=100, bbox_inches='tight')
plt.close()

# Prepare data for modeling
df_model = pd.get_dummies(df[['Sales', 'Region', 'Segment', 'Category']], drop_first=True)
X = df_model.drop('Sales', axis=1)
y = df_model['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Chart 5: Actual vs Predicted
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Sales', fontweight='bold')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.tight_layout()
plt.savefig('images/05_actual_vs_predicted.png', dpi=100, bbox_inches='tight')
plt.close()

# Chart 6: Residual Distribution
plt.figure(figsize=(10,6))
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution', fontweight='bold')
plt.tight_layout()
plt.savefig('images/06_residual_distribution.png', dpi=100, bbox_inches='tight')
plt.close()

print('All charts saved successfully!')
