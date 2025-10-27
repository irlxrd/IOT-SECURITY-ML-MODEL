import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
import joblib
import os
from itertools import cycle

# Create directories
os.makedirs('web_app', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Load the dataset
data = pd.read_csv('balanced_multiclass.csv')

# Define features and target
X = data.drop(columns=['label'])
y = data['label']

# Encode categorical features
X = pd.get_dummies(X, columns=['proto', 'state'], drop_first=True)

# Replace '-' with NaN and handle missing values
X.replace('-', pd.NA, inplace=True)
X = X.apply(pd.to_numeric, errors='ignore')
X.fillna(X.median(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("="*80)
print("TRAINING MODELS")
print("="*80)

# --------Train a Random Forest model----------
print("\nTraining Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print accuracy scores
rf_accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# Extract feature importance
importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance_df.head(10))

# ----------Train a Gradient Boosting model----------
print("\nTraining Gradient Boosting...")
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train, y_train)

# Predict and evaluate Gradient Boosting model
gb_y_pred = gb_clf.predict(X_test)
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, gb_y_pred))
print("Gradient Boosting Confusion Matrix:")
print(confusion_matrix(y_test, gb_y_pred))

# Print accuracy scores
gb_accuracy = accuracy_score(y_test, gb_y_pred)
print(f"Gradient Boosting Accuracy: {gb_accuracy:.2f}")

# ----------XGBoost section - Add Label Encoding----------
print("\nTraining XGBoost...")
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss')

# Convert string labels to numerical labels for XGBoost
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

xgb.fit(X_train, y_train_encoded)
y_pred_xgb_encoded = xgb.predict(X_test)
y_pred_xgb = le.inverse_transform(y_pred_xgb_encoded)  # Convert back to string labels for reporting

print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")

# ----------Train a simple Neural Network (MLP) with scaling----------
print("\nTraining MLP...")
scaler = StandardScaler()

# Fit scaler on training data and transform both train/test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Predict and evaluate MLP
y_pred_mlp = mlp.predict(X_test_scaled)
print("\nMLP Classification Report:")
print(classification_report(y_test, y_pred_mlp))
print("MLP Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_mlp))
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
print(f"MLP Accuracy: {mlp_accuracy:.2f}")

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. Feature Importance Comparison
print("\n1. Creating feature importance comparison plot...")
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

top_n = 15
rf_imp = feature_importance_df.head(top_n)
axes[0, 0].barh(rf_imp['Feature'], rf_imp['Importance'], color='#1f77b4')
axes[0, 0].set_title('Random Forest - Top Feature Importances', fontsize=14, fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].set_xlabel('Importance')

# Gradient Boosting
gb_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb_clf.feature_importances_
}).sort_values(by='Importance', ascending=False).head(top_n)
axes[0, 1].barh(gb_importances['Feature'], gb_importances['Importance'], color='#ff7f0e')
axes[0, 1].set_title('Gradient Boosting - Top Feature Importances', fontsize=14, fontweight='bold')
axes[0, 1].invert_yaxis()
axes[0, 1].set_xlabel('Importance')

# XGBoost
xgb_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb.feature_importances_
}).sort_values(by='Importance', ascending=False).head(top_n)
axes[1, 0].barh(xgb_importances['Feature'], xgb_importances['Importance'], color='#2ca02c')
axes[1, 0].set_title('XGBoost - Top Feature Importances', fontsize=14, fontweight='bold')
axes[1, 0].invert_yaxis()
axes[1, 0].set_xlabel('Importance')

axes[1, 1].axis('off')  # Hide the fourth subplot
plt.tight_layout()
plt.savefig('plots/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Model Accuracy Comparison
print("2. Creating model accuracy comparison plot...")
accuracies = {
    'Random Forest': rf_accuracy,
    'Gradient Boosting': gb_accuracy,
    'XGBoost': xgb_accuracy,
    'MLP': mlp_accuracy
}

plt.figure(figsize=(10, 6))
bars = plt.bar(accuracies.keys(), accuracies.values(), 
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.savefig('plots/model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Confusion Matrix Heatmaps
print("3. Creating confusion matrix heatmaps...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

models = [
    ('Random Forest', y_pred),
    ('Gradient Boosting', gb_y_pred),
    ('XGBoost', y_pred_xgb),
    ('MLP', y_pred_mlp)
]

for idx, (name, predictions) in enumerate(models):
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clf.classes_, yticklabels=clf.classes_,
                ax=axes[idx // 2, idx % 2], cbar_kws={'label': 'Count'})
    axes[idx // 2, idx % 2].set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
    axes[idx // 2, idx % 2].set_ylabel('True Label', fontsize=10)
    axes[idx // 2, idx % 2].set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Per-Class Performance Metrics
print("4. Creating per-class performance metrics plot...")
metrics_data = []
for name, predictions in models:
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average=None)
    for idx, class_name in enumerate(clf.classes_):
        metrics_data.append({
            'Model': name,
            'Class': class_name,
            'Precision': precision[idx],
            'Recall': recall[idx],
            'F1-Score': f1[idx]
        })

metrics_df = pd.DataFrame(metrics_data)

# Plot grouped bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = ['Precision', 'Recall', 'F1-Score']

for idx, metric in enumerate(metrics):
    pivot_df = metrics_df.pivot(index='Class', columns='Model', values=metric)
    pivot_df.plot(kind='bar', ax=axes[idx])
    axes[idx].set_title(f'{metric} by Class', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(metric, fontsize=10)
    axes[idx].set_xlabel('Class', fontsize=10)
    axes[idx].legend(title='Model', loc='lower right')
    axes[idx].set_ylim([0, 1.1])
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/per_class_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Learning Curve for Random Forest
print("5. Creating learning curve plot...")
def plot_learning_curve(estimator, title, X, y, cv=5):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, label='Cross-validation score', color='red', linewidth=2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'plots/learning_curve_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_learning_curve(clf, 'Random Forest Learning Curve', X_train, y_train)

# 6. ROC Curves (Multiclass)
print("6. Creating ROC curves plot...")
# Binarize labels for multiclass ROC
y_test_bin = label_binarize(y_test, classes=clf.classes_)
n_classes = y_test_bin.shape[1]

# Get probability predictions
y_score = clf.predict_proba(X_test)

# Compute ROC curve for each class
plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'])
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{clf.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Random Forest (One-vs-Rest)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.3)
plt.savefig('plots/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Decision Tree Visualization
print("7. Creating decision tree visualization...")
estimator = clf.estimators_[0]
plt.figure(figsize=(20, 10))
plot_tree(estimator, feature_names=X.columns, class_names=clf.classes_, 
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree from Random Forest", fontsize=16, fontweight='bold')
plt.savefig('plots/decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

# Save models
joblib.dump(clf, 'web_app/random_forest_model.pkl')
joblib.dump(gb_clf, 'web_app/gradient_boosting_model.pkl')
joblib.dump(xgb, 'web_app/xgb_model.pkl')
joblib.dump(le, 'web_app/label_encoder.pkl')
joblib.dump(mlp, 'web_app/mlp_model.pkl')
joblib.dump(scaler, 'web_app/scaler.pkl')

print("\nAll models saved successfully!")
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nModel Accuracies:")
print(f"  - Random Forest:      {rf_accuracy:.4f}")
print(f"  - Gradient Boosting:  {gb_accuracy:.4f}")
print(f"  - XGBoost:            {xgb_accuracy:.4f}")
print(f"  - MLP:                {mlp_accuracy:.4f}")
print(f"\nAll visualizations saved in 'plots/' directory")
print(f"All models saved in 'web_app/' directory")
print("\n" + "="*80)
