import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

X_train = pd.read_csv('./dataset/X_train.csv')
y_train = pd.read_csv('./dataset/y_train.csv')
X_test = pd.read_csv('./dataset/X_test.csv')
y_test = pd.read_csv('./dataset/y_test.csv')

pipeline = Pipeline([('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(random_state=11,max_iter=1000))
                    ])

stratified_kfold = StratifiedKFold(n_splits=3,
                                       shuffle=True,
                                       random_state=11)
    
param_grid = {'classifier__C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=stratified_kfold,
                           n_jobs=-1)


grid_search.fit(X_train, y_train.values.ravel())

best_model = grid_search.best_estimator_

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f'Best Hyperparameters: {grid_search.best_params_}')
print(f'Train Accuracy: {train_acc:.4f}')
print(f'Best Cross-validation Accuracy: {grid_search.best_score_:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
print(f'F1 Score (Test): {f1:.4f}')