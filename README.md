
1#fraud_detection
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# 1. Generate Synthetic Financial Transaction Data
np.random.seed(42)
n_records = 1000

data = {
    'transaction_id': range(1001, 1001 + n_records),
    'amount': np.random.exponential(scale=100, size=n_records),
    'transaction_hour': np.random.randint(0, 24, size=n_records),
    'merchant_category': np.random.choice(['Grocery', 'Electronics', 'Food', 'Travel', 'Clothing'], size=n_records),
    'foreign_transaction': np.random.choice([0, 1], size=n_records, p=[0.9, 0.1])
}

df = pd.DataFrame(data)

# Inject synthetic anomaly/fraud cases
df.loc[15, 'amount'] = 8500.00
df.loc[42, 'amount'] = 9200.50
df.loc[108, 'amount'] = 6700.00

# 2. Anomaly Detection Engine
features = df[['amount', 'transaction_hour', 'foreign_transaction']]

model = IsolationForest(contamination=0.005, random_state=42)
df['anomaly_score'] = model.fit_predict(features)

# Isolation Forest tags outliers as -1
df['is_flagged_fraud'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

# 3. Export High-Risk Audit Output
flagged_records = df[df['is_flagged_fraud'] == 1]
flagged_records.to_csv("flagged_fraud_audit.csv", index=False)

print(f"Audit Complete: {len(flagged_records)} suspicious transactions flagged and saved to 'flagged_fraud_audit.csv'.")



2# high_risk_attrition_report.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate Synthetic HR Dataset
np.random.seed(42)
n_employees = 500

data = {
    'employee_id': range(101, 101 + n_employees),
    'age': np.random.randint(22, 60, size=n_employees),
    'monthly_income': np.random.randint(2500, 15000, size=n_employees),
    'years_at_company': np.random.randint(1, 15, size=n_employees),
    'overtime': np.random.choice([0, 1], size=n_employees, p=[0.7, 0.3]),
    'job_satisfaction': np.random.randint(1, 5, size=n_employees), # 1 (Low) to 4 (High)
    'attrition': np.random.choice([0, 1], size=n_employees, p=[0.82, 0.18])
}

df = pd.DataFrame(data)

# 2. Predictive Modeling
X = df[['age', 'monthly_income', 'years_at_company', 'overtime', 'job_satisfaction']]
y = df['attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# 3. Generate Summary Results
df['attrition_risk_score'] = clf.predict_proba(X)[:, 1]
high_risk_employees = df[df['attrition_risk_score'] > 0.6]

high_risk_employees.to_csv("high_risk_attrition_report.csv", index=False)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"HR Alert: {len(high_risk_employees)} employees flagged as high attrition risk.")