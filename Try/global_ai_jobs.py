import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────
# 1️⃣ LOAD DATASET
# ─────────────────────────────────────────────
df = pd.read_csv('global_ai_jobs.csv')

print("=" * 60)
print("📦 DATASET INFORMATION")
print("=" * 60)
print("Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# ─────────────────────────────────────────────
# 2️⃣ CLEAN DATA
# ─────────────────────────────────────────────
print("\n🔧 Cleaning Missing Values...")

# Fill numeric columns with median
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Remaining Missing Values:", df.isnull().sum().sum())

# ─────────────────────────────────────────────
# 3️⃣ VISUALIZATION
# ─────────────────────────────────────────────
plt.figure(figsize=(18, 12))

plt.subplot(2, 2, 1)
df['country'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Countries")

plt.subplot(2, 2, 2)
sns.histplot(df['salary_usd'], bins=40)
plt.title("Salary Distribution")

plt.subplot(2, 2, 3)
df['job_role'].value_counts().head(8).plot(kind='bar')
plt.title("Top Job Roles")

plt.subplot(2, 2, 4)
sns.boxplot(x='experience_level', y='salary_usd', data=df)
plt.title("Salary by Experience")

plt.tight_layout()
plt.savefig("ai_jobs_visualization.png")
plt.show()

print("✅ Visualization saved!")

# ─────────────────────────────────────────────
# 4️⃣ CREATE TARGET VARIABLE
# ─────────────────────────────────────────────
median_salary = df['salary_usd'].median()
df['high_salary'] = (df['salary_usd'] > median_salary).astype(int)

print(f"\nMedian Salary: ${median_salary:,.0f}")
print("High Salary Count:", df['high_salary'].sum())

# ─────────────────────────────────────────────
# 5️⃣ FEATURE SELECTION
# ─────────────────────────────────────────────

numeric_cols = [
    'experience_years', 'bonus_usd', 'interview_rounds',
    'weekly_hours', 'company_rating', 'job_openings',
    'hiring_difficulty_score', 'layoff_risk', 'ai_adoption_score',
    'company_funding_billion', 'economic_index', 'ai_maturity_years',
    'offer_acceptance_rate', 'tax_rate_percent', 'vacation_days',
    'skill_demand_score', 'automation_risk', 'job_security_score',
    'career_growth_score', 'work_life_balance_score',
    'promotion_speed', 'cost_of_living_index', 'employee_satisfaction'
]

categorical_cols = ['country', 'job_role', 'work_mode', 'experience_level']

# One-hot encoding
df_encoded = pd.get_dummies(
    df[numeric_cols + categorical_cols],
    drop_first=True
)

X = df_encoded
y = df['high_salary']

# ─────────────────────────────────────────────
# 6️⃣ TRAIN-TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ─────────────────────────────────────────────
# 7️⃣ TRAIN MODEL
# ─────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 8️⃣ EVALUATION
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

train_accuracy = model.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 TRAINING ACCURACY:", round(train_accuracy * 100, 2), "%")
print("🎯 TESTING ACCURACY:", round(test_accuracy * 100, 2), "%")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────
# 9️⃣ CROSS VALIDATION
# ─────────────────────────────────────────────
cv_scores = cross_val_score(model, X, y, cv=5)

print("\n🔁 Cross Validation Scores:", cv_scores)
print("📊 Average CV Accuracy:", round(cv_scores.mean() * 100, 2), "%")

# ─────────────────────────────────────────────
# 🔟 FEATURE IMPORTANCE
# ─────────────────────────────────────────────
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n🔑 Top 10 Important Features:")
print(importance.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=importance.head(10)
)
plt.title("Top 10 Features Predicting High Salary")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ─────────────────────────────────────────────
# 1️⃣1️⃣ SAVE MODEL
# ─────────────────────────────────────────────
with open("ai_jobs_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved as ai_jobs_model.pkl")
print("\n🎉 PROJECT COMPLETE SUCCESSFULLY!")