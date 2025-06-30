

# task_5_decision_tree_random_forest.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("heart.csv")

# Features and labels
X = df.drop('target', axis=1)
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree (Pruned)
dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_pruned.fit(X_train, y_train)
y_pred_dt = dt_pruned.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Plot Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dt_pruned, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'])
plt.title("Decision Tree (Max Depth = 4)")
plt.savefig("decision_tree.png")
plt.close()

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Feature Importances
importances = rf.feature_importances_
feat_importances = pd.Series(importances, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importances - Random Forest")
plt.savefig("feature_importance.png")
plt.close()

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)
print("Cross-validation Accuracy: %.2f%% (+/- %.2f%%)" % (cv_scores.mean()*100, cv_scores.std()*100))

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
