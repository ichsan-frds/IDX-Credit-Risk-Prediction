import pandas as pd
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train = pd.read_csv('./dataset/X_train.csv')
y_train = pd.read_csv('./dataset/y_train.csv')
X_test = pd.read_csv('./dataset/X_test.csv')
y_test = pd.read_csv('./dataset/y_test.csv')

pipeline = imbpipeline(steps = [
    ['smote', SMOTE(random_state=11)],
    ['scaler', MinMaxScaler()],
    ['classifier', RandomForestClassifier(random_state=11, max_depth=20, n_estimators=200,min_samples_split=10,min_samples_leaf=4)]
])

pipeline.fit(X_train, y_train.values.ravel())

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f'Train Accuracy: {train_acc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
print(f'F1 Score (Test): {f1:.4f}')