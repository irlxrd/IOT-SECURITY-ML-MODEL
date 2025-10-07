import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

def train_models(csv_file):
    """
    Train models on preprocessed CSV data
    """
    print(f"Loading preprocessed data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows")
    
    # Show label distribution
    if "label" in df.columns:
        print("\nLabel distribution:")
        print(df["label"].value_counts())
        
        # Encode labels
        df["label_enc"], uniques = pd.factorize(df["label"])
        print("\nLabel encoding mapping:")
        for idx, label in enumerate(uniques):
            print(f"{label}: {idx}")
    
    # One-hot encode categorical features
    cat_cols = ["proto", "state"]
    df = pd.get_dummies(df, columns=cat_cols)
    
    # Prepare features and target
    X = df.drop(['label', 'label_enc'], axis=1)
    y = df['label_enc']
    
    print(f"\nFeature columns: {X.columns.tolist()}")
    print(f"Dataset shape: {X.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_train, y_train)
    lr_accuracy = lr.score(X_test, y_test)
    print(f"Logistic Regression accuracy: {lr_accuracy:.4f}")
    
    # Feature importance for Logistic Regression (multi-class)
    if lr.coef_.shape[0] > 1:  # Multi-class
        # Use mean absolute coefficient across all classes
        mean_coef = np.mean(np.abs(lr.coef_), axis=0)
        importances_lr = pd.DataFrame({
            'feature': X.columns, 
            'mean_abs_coefficient': mean_coef
        })
        print("\nTop 10 Logistic Regression feature importances (mean absolute coefficient):")
        print(importances_lr.sort_values('mean_abs_coefficient', ascending=False).head(10))
    else:  # Binary classification
        importances_lr = pd.DataFrame({
            'feature': X.columns, 
            'coefficient': lr.coef_[0]
        })
        print("\nTop 10 Logistic Regression feature importances:")
        print(importances_lr.sort_values('coefficient', key=abs, ascending=False).head(10))
    
    # Train Decision Tree
    print("\nTraining Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    dt_accuracy = dt.score(X_test, y_test)
    print(f"Decision Tree accuracy: {dt_accuracy:.4f}")
    
    # Feature importance for Decision Tree
    importances_dt = pd.DataFrame({
        'feature': X.columns, 
        'importance': dt.feature_importances_
    })
    print("\nTop 10 Decision Tree feature importances:")
    print(importances_dt.sort_values('importance', ascending=False).head(10))
    
    # Visualize Decision Tree
    plt.figure(figsize=(16, 8))
    plot_tree(dt, feature_names=X.columns, 
              class_names=[str(c) for c in y.unique()], 
              filled=True, max_depth=3)
    plt.title("Decision Tree Visualization (max_depth=3)")
    plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save models and results
    import joblib
    print("\nSaving trained models...")
    joblib.dump(lr, 'logistic_regression_model.pkl')
    joblib.dump(dt, 'decision_tree_model.pkl')
    
    # Save label mapping
    label_mapping = {idx: label for idx, label in enumerate(uniques)}
    joblib.dump(label_mapping, 'label_mapping.pkl')
    
    # Save feature columns for future predictions
    joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
    
    print("✓ Saved logistic_regression_model.pkl")
    print("✓ Saved decision_tree_model.pkl") 
    print("✓ Saved label_mapping.pkl")
    print("✓ Saved feature_columns.pkl")
    print("✓ Saved decision_tree_visualization.png")
    
    return lr, dt, X_test, y_test

if __name__ == "__main__":
    # Train on combined dataset
    lr_model, dt_model, X_test, y_test = train_models("combined_clean.csv")