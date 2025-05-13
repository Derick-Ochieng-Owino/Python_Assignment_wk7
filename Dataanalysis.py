import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Generate sample data
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
categories = ['Electronics', 'Clothing', 'Home Goods', 'Groceries']

data = {
    'Date': np.random.choice(dates, 500),
    'Category': np.random.choice(categories, 500),
    'Sales': np.abs(np.random.normal(1000, 300, 500)),
    'Units': np.random.randint(1, 25, 500),
    'Rating': np.random.uniform(2, 5, 500).round(1)
}

df = pd.DataFrame(data)
df['Month'] = df['Date'].dt.month_name()

sales_df['Customer Rating'] = sales_df['Customer Rating'].fillna(
    sales_df['Customer Rating'].median())

sales_df['Month'] = sales_df['Date'].dt.month_name()

print("Sales Amount Statistics:")
print(sales_df['Sales Amount'].describe())

print("\nAverage Sales by Region:")
print(sales_df.groupby('Region')['Sales Amount'].mean().round(2))

print("\nTotal Units by Category:")
print(sales_df.groupby('Category')['Units Sold'].sum())


plt.figure(figsize=(12, 5))
monthly_trend = sales_df.groupby(sales_df['Date'].dt.month)['Sales Amount'].sum()
monthly_trend.plot(kind='line', marker='o', color='teal', linewidth=2)
plt.title('Monthly Sales Trend 2023', pad=20, fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales ($)', fontsize=12)
plt.xticks(range(1,13), ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 5))
category_avg = df.groupby('Category')['Sales'].mean().sort_values()

sns.barplot(x=category_avg.index, y=category_avg.values, 
            palette='viridis', alpha=0.8)
plt.title('Average Sales by Category', fontsize=14, pad=15)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Average Sales ($)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 5))
sns.histplot(df['Sales'], bins=15, kde=True, 
             color='teal', alpha=0.6)
plt.title('Sales Amount Distribution', fontsize=14, pad=15)
plt.xlabel('Sales Amount ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Units', y='Sales', 
                hue='Category', palette='Set2',
                alpha=0.7, s=100, edgecolor='white')

plt.title('Sales vs Units Sold by Category', fontsize=14, pad=15)
plt.xlabel('Units Sold', fontsize=12)
plt.ylabel('Sales Amount ($)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)

# Add jitter to reduce overlap
plt.xlim(df['Units'].min()-0.5, df['Units'].max()+0.5)
plt.tight_layout()
plt.show()
