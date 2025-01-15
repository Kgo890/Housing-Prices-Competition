import numpy as np
from matplotlib import pyplot as plt

from data import training_dataset, testing_dataset, y_training, X_training, X_testing, preprocessor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, root_mean_squared_error, r2_score, \
    mean_squared_error

# Splitting the both datasets into 80% will be used for training and the 20% will be used for the validation dataset
X_train, X_validation, y_train, y_validation = train_test_split(X_training, y_training, test_size=0.2, random_state=1)
'''
# SpotChecking Algorithms to see which one performs the best
models = [("LR", LinearRegression()), ("GB", GradientBoostingRegressor()), ("RF", RandomForestRegressor())]

results = []
names = []
print("Evaluating Models using kfold and pipeline")
# evaluating the models using pipeline and kfold
for name, model in models:
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    kfold = KFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring="r2")
    results.append(cv_results)
    names.append(name)
    print(f"Model: {name}")
    print(f"R² Score: Mean={cv_results.mean():.4f}, Std={cv_results.std():.4f}")

# The one that performed better is Gradient Boosting
# Making a box and whisker plot to compare algorithms
plt.boxplot(results)
plt.title("Algorithms Comparison")
plt.show()

'''

print("Using Gradient Boosting to make predictions")
# Making predictions on a validation set using gradient boosting
best_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, loss="huber", random_state=10)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_validation)
# making sure that prediction don't have negatives
predictions = np.maximum(predictions, 0)

# Evaluating the predictions
print("R2 score")
print(r2_score(y_validation, predictions))
print("root mean squared error of the predictions")
rmse = root_mean_squared_error(y_validation, predictions)
print("$" + str(rmse))

# making predictions on the testing dataset
test_predictions = pipeline.predict(X_testing)
test_predictions = np.maximum(test_predictions, 0)

# submission file
submission = testing_dataset[["Id"]].copy()
submission["SalePrice"] = test_predictions
submission.to_csv("submission.csv", index=False)
print("Submisssion file has been created: submission.csv")
