import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add the project root to sys.path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.features.vectorizer import train_vectorizer, save_vectorizer

def load_data(path="data/processed/processed_dataset.csv"):
    """
    Loads the processed dataset from the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    print(f"Loading processed dataset from {path}...")
    return pd.read_csv(path)

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluates the model and prints accuracy, precision, recall, f1-score, and confusion matrix.
    Returns the f1-score to help in deciding the best model.
    """
    print(f"\n--- Evaluation for {model_name} ---")
    
    # Using weighted average to handle multi-class classification
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    return f1

def main():
    # 1. Load processed dataset
    df = load_data()
    
    # Drop rows with missing values if any
    df = df.dropna(subset=['combined_metadata', 'function_name'])
    
    X_text = df['combined_metadata'].astype(str)
    y = df['function_name']
    
    # 2. Train TF-IDF vectorizer
    X_vec = train_vectorizer(X_text)
    
    # Save vectorizer
    save_vectorizer(os.path.join("models", "vectorizer.pkl"))
    
    # 3. Split data into train/test (80/20)
    print("\nSplitting data into train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # 4. Train Models
    print("\nTraining models...")
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_f1 = evaluate_model(y_test, lr_preds, "Logistic Regression")
    
    # Multinomial Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)
    nb_f1 = evaluate_model(y_test, nb_preds, "Multinomial Naive Bayes")
    
    # 5. Select Best Model
    print("\n==================================")
    print("Model Selection:")
    
    best_model = None
    best_name = ""
    
    if lr_f1 >= nb_f1:
        best_model = lr_model
        best_name = "Logistic Regression"
    else:
        best_model = nb_model
        best_name = "Multinomial Naive Bayes"
        
    print(f"Best model selected: {best_name} (F1 Score: {max(lr_f1, nb_f1):.4f})")
    print("==================================")
    
    # 6. Save the best model
    model_path = os.path.join("models", "function_model.pkl")
    print(f"\nSaving best model to {model_path}...")
    joblib.dump(best_model, model_path)
    print("Model saved successfully. Training pipeline complete.")

if __name__ == "__main__":
    main()
