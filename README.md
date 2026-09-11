# high_risk_attrition_report.py
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