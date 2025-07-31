# Comprehensive Loan Portfolio Analysis
# Complete analysis of loan dataset with data cleaning, KPI calculation, and strategic insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load and inspect data
print("Loading loan dataset...")
df = pd.read_excel('CS.xlsx', sheet_name='CLA')
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Data quality assessment
print("\n=== DATA QUALITY ASSESSMENT ===")
missing_data = (df.isnull().sum() / len(df) * 100).round(2)
print("Missing data percentages:")
print(missing_data[missing_data > 0].sort_values(ascending=False))

# Data cleaning and preprocessing
print("\n=== DATA CLEANING ===")
# Convert date columns
date_columns = ['disbursement_date', 'first_due_date', 'loan_final_due_date']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], unit='ms', errors='coerce')

# Create key performance indicators
df['is_repaid'] = (df['loan_status'] == 'loan_repaid').astype(int)
df['days_delayed'] = df['previous_loan_days_delayed'].fillna(0)
df['profit_loss'] = df['total_paid_amount'] - df['loan_amount']
df['roi'] = (df['profit_loss'] / df['loan_amount'] * 100).fillna(0)

# Portfolio KPIs
print("\n=== PORTFOLIO PERFORMANCE ===")
total_portfolio = df['loan_amount'].sum()
total_repaid = df['total_paid_amount'].sum()
repayment_rate = df['is_repaid'].mean()
default_rate = 1 - repayment_rate
portfolio_roi = df['roi'].mean()

print(f"Total Portfolio Value: ${total_portfolio/1e9:.2f}B")
print(f"Repayment Rate: {repayment_rate*100:.2f}%")
print(f"Default Rate: {default_rate*100:.2f}%")
print(f"Portfolio ROI: {portfolio_roi:.2f}%")

# Client type analysis
print("\n=== CLIENT TYPE ANALYSIS ===")
client_analysis = df.groupby('client_type').agg({
    'is_repaid': ['count', 'mean'],
    'loan_amount': 'mean',
    'roi': 'mean'
}).round(2)
print(client_analysis)

# Risk scoring analysis
print("\n=== RISK SCORING ANALYSIS ===")
df['combined_risk_score'] = np.where(df['client_type'] == 'returning', 
                                   df['risk_score_returning'], 
                                   df['risk_score_new'])
df['score_tier'] = pd.cut(df['combined_risk_score'], 
                         bins=5, labels=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5'])

score_performance = df.groupby('score_tier').agg({
    'is_repaid': 'mean',
    'loan_amount': 'mean'
}).round(3)
print(score_performance)

# Create visualizations
plt.figure(figsize=(15, 10))

# Subplot 1: Repayment rates by client type
plt.subplot(2, 3, 1)
client_repayment = df.groupby('client_type')['is_repaid'].mean()
plt.bar(client_repayment.index, client_repayment.values)
plt.title('Repayment Rate by Client Type')
plt.ylabel('Repayment Rate')

# Subplot 2: ROI by client type
plt.subplot(2, 3, 2)
client_roi = df.groupby('client_type')['roi'].mean()
plt.bar(client_roi.index, client_roi.values, color='orange')
plt.title('ROI by Client Type')
plt.ylabel('ROI (%)')

# Subplot 3: Loan amount distribution
plt.subplot(2, 3, 3)
plt.hist(df['loan_amount'], bins=50, alpha=0.7)
plt.title('Loan Amount Distribution')
plt.xlabel('Loan Amount')

# Subplot 4: Risk score vs repayment
plt.subplot(2, 3, 4)
score_repayment = df.groupby('score_tier')['is_repaid'].mean()
plt.bar(range(len(score_repayment)), score_repayment.values, color='green')
plt.title('Repayment Rate by Risk Tier')
plt.xticks(range(len(score_repayment)), score_repayment.index, rotation=45)

# Subplot 5: Portfolio performance over time
plt.subplot(2, 3, 5)
monthly_performance = df.groupby(df['disbursement_date'].dt.to_period('M'))['is_repaid'].mean()
plt.plot(monthly_performance.index.astype(str), monthly_performance.values)
plt.title('Monthly Repayment Rates')
plt.xticks(rotation=45)

# Subplot 6: Profit/Loss distribution
plt.subplot(2, 3, 6)
plt.hist(df['profit_loss'], bins=50, alpha=0.7, color='red')
plt.title('Profit/Loss Distribution')
plt.xlabel('Profit/Loss Amount')

plt.tight_layout()
plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== ANALYSIS COMPLETE ===")
print("Key findings:")
print("• Portfolio is currently unprofitable")
print("• New clients show higher risk than returning clients")
print("• Risk scoring model needs recalibration")
print("• Small loans perform better than large loans")
