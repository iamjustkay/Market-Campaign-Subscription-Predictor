import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pylab import rcParams
import seaborn as sns
import warnings
sns.set_style('darkgrid')
rcParams['figure.figsize'] = 8,8
#%matplotlib inline
warnings.filterwarnings("ignore")
import joblib

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import ADASYN

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Load files
train = pd.read_csv('bank-full.csv', sep=";")
test = pd.read_csv('bank.csv', sep=";")

# Normalise Train Numerical Attributes

from sklearn.preprocessing import MinMaxScaler

attributes_to_normalize = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
data_to_normalize = train[attributes_to_normalize]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data_to_normalize)
train[attributes_to_normalize] = normalized_data

train[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']].describe()

# Normalise Test Numerical Attributes


attributes_to_normalize = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
data_to_normalize = test[attributes_to_normalize]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data_to_normalize)
test[attributes_to_normalize] = normalized_data

test[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']].describe()

# Train Target conversion
train['y'] = train['y'].map({'yes': 1, 'no': 0})

# Convert month to ordinal
month_order = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
               'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
train['month'] = train['month'].map(month_order)

# New feature: previously contacted
train['contacted_before'] = train['pdays'].apply(lambda x: 0 if x == -1 else 1)

# Test Target conversion
test['y'] = test['y'].map({'yes': 1, 'no': 0})

# Convert month to ordinal
month_order = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
               'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
test['month'] = test['month'].map(month_order)

# New feature: previously contacted
test['contacted_before'] = test['pdays'].apply(lambda x: 0 if x == -1 else 1)

X_train = train.drop(columns='y', axis=1)
y_train = train['y']

print('Training Data Set Shape:', X_train.shape, y_train.shape)

# Identify columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()


# Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

adasyn = ADASYN(random_state=42, sampling_strategy=0.8)

# Apply the preprocessor to X_train before resampling
X_train_processed = preprocessor.fit_transform(X_train)

X_resampled, y_resampled = adasyn.fit_resample(X_train_processed, y_train)

# The number of columns in X_resampled will be different from X_train due to one-hot encoding,
# so we cannot use X_train.columns here.
# train_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name='y')], axis=1)

print('Resampled Data Set Shape:', X_resampled.shape, y_resampled.shape)
print(y_resampled.value_counts())

# Get feature names from preprocessor
feature_names = preprocessor.get_feature_names_out()

# Create DataFrame from resampled data
X_df_resampled = pd.DataFrame(X_resampled, columns=feature_names)
X_df_resampled['y'] = y_resampled

# Compute correlations with the target
correlations = X_df_resampled.corr()['y'].drop('y').sort_values(ascending=False)

# Plot correlation bar chart
plt.figure(figsize=(10, 8))
sns.barplot(x=correlations.values, y=correlations.index, palette='viridis')
plt.title('Feature Correlation with Target (y)')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.grid(True)
plt.show()

# Define model pipelines
models = {
    "Logistic Regression": Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(class_weight='balanced', random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ]),
    "XGBoost": Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=7, random_state=42))
    ])
}

X_test = test.drop(columns='y', axis=1)
y_test = test['y']

print('Test Data Set Shape:', X_test.shape, y_test.shape)

# Before preprocessing
assert isinstance(X_train, pd.DataFrame)
assert isinstance(X_test, pd.DataFrame)

# Apply preprocessor safely
preprocessor.fit(X_train)  # only fit on original training data
X_train_processed = preprocessor.transform(X_train)  # transform after fit
X_test_processed = preprocessor.transform(X_test)    # safe to apply now



# Define models (no preprocessing inside)
models_resampled = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and evaluate
results_resampled = []

for name, model in models_resampled.items():
    print(f"\n🧪 Retraining on ADASYN data: {name}")
    
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    results_resampled.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'ROC-AUC': roc
    })

    print(classification_report(y_test, y_pred))


df_resampled_results = pd.DataFrame(results_resampled).sort_values(by='F1 Score', ascending=False)
print("\n📋 Resampled Model Performance:")
print(df_resampled_results)

results = []

for name, model in models.items():
    print(f"\n🧪 Evaluating: {name}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'ROC-AUC': roc
    })

    print(classification_report(y_test, y_pred))

df_results = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
print("\n📋 Model Performance:")
print(df_results)

# Save both preprocessor and the best model (e.g., XGBoost)
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(models_resampled["Random Forest"], "RF_model.pkl", compress=3)  # or replace with Logistic Regression, etc.