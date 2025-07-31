
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('CS.xlsx', sheet_name='CLA')

missing_analysis = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_analysis,
    'Missing_Percentage': missing_pct
}).sort_values('Missing_Percentage', ascending=False)

date_columns = ['disbursement_date', 'first_due_date', 'original_first_due_date', 
                'loan_final_due_date', 'max_loan_payment_date']

def convert_date_column(col):
    if col.dtype == 'object':
        return pd.to_datetime(col, errors='coerce')
    else:
        return pd.to_datetime(col, unit='ms', errors='coerce')

for col in date_columns:
    if col in df.columns:
        df[col + '_converted'] = convert_date_column(df[col])

if 'disbursement_date_converted' in df.columns and 'first_due_date_converted' in df.columns:
    invalid_dates = df[df['disbursement_date_converted'] > df['first_due_date_converted']].shape[0]

duplicates = df.duplicated(subset=['user_id', 'loan_amount', 'disbursement_date']).sum()

numeric_cols = ['loan_amount', 'disbursement_amount', 'declared_income', 'previously_paid_loans']
outlier_counts = {}
for col in numeric_cols:
    if col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        outlier_counts[col] = outliers

on_time_payments = df[df['first_instalment_status'] == 'paid'].shape[0]
on_time_rate = (on_time_payments / len(df)) * 100

defaults = df[df['loan_status'] == 'loan_ongoing'].shape[0]
default_rate = (defaults / len(df)) * 100

returning_customers = df[df['client_type'] == 'returning'].copy()
returning_customers['risk_quartile'] = pd.qcut(returning_customers['risk_score_returning'].dropna(), 
                                               q=4, labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High'])

score_performance = returning_customers.groupby('risk_quartile').agg({
    'loan_status': lambda x: (x == 'loan_repaid').mean() * 100,
    'user_id': 'count'
}).round(2)
score_performance.columns = ['Repayment_Rate_%', 'Count']

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
default_by_type = df.groupby('client_type')['loan_status'].apply(lambda x: (x == 'loan_ongoing').mean() * 100)
default_by_type.plot(kind='bar')
plt.title('Default Rate by Client Type')
plt.ylabel('Default Rate (%)')
plt.xticks(rotation=45)

plt.subplot(2, 3, 2)
df['loan_amount'].hist(bins=50, alpha=0.7)
plt.title('Loan Amount Distribution')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')

plt.subplot(2, 3, 3)
if not returning_customers.empty:
    score_perf_plot = returning_customers.groupby('risk_quartile')['loan_status'].apply(lambda x: (x == 'loan_repaid').mean() * 100)
    score_perf_plot.plot(kind='bar')
    plt.title('Repayment Rate by Risk Quartile')
    plt.ylabel('Repayment Rate (%)')
    plt.xticks(rotation=45)

plt.subplot(2, 3, 4)
geo_performance = df.groupby('location_state')['loan_status'].apply(lambda x: (x == 'loan_repaid').mean() * 100).sort_values(ascending=False).head(10)
geo_performance.plot(kind='bar')
plt.title('Top 10 States by Repayment Rate')
plt.ylabel('Repayment Rate (%)')
plt.xticks(rotation=45)

plt.subplot(2, 3, 5)
df['disbursement_month'] = pd.to_datetime(df['disbursement_date_converted']).dt.to_period('M')
monthly_trend = df.groupby('disbursement_month')['loan_status'].apply(lambda x: (x == 'loan_repaid').mean() * 100)
monthly_trend.plot()
plt.title('Monthly Repayment Rate Trend')
plt.ylabel('Repayment Rate (%)')
plt.xticks(rotation=45)

plt.subplot(2, 3, 6)
df['amount_bucket'] = pd.cut(df['loan_amount'], bins=10)
amount_default = df.groupby('amount_bucket')['loan_status'].apply(lambda x: (x == 'loan_ongoing').mean() * 100)
amount_default.plot(kind='bar')
plt.title('Default Rate by Loan Amount')
plt.ylabel('Default Rate (%)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('loan_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
