import Decision_Trees as dt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd


##### Data Setup ######
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
data = load_breast_cancer()
X = data.data           
y = data.target        
feature_names = data.feature_names  
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

class RandomForest:
    def __init__(self,min_sample_split=2 , max_depth=4 , n_trees=10, n_features=None):
        self.min_sample_split=min_sample_split
        self.max_depth=max_depth
        self.n_trees=n_trees
        self.n_features=n_features
        self.trees=[]
    
    def fit(self,X,y):
        self.trees=[]
        for _ in range(self.n_trees):
            tree=dt.DecisionTree(min_samples_split=self.min_sample_split, max_depth=self.max_depth, max_features=self.n_features)
            X_bt, y_bt=self._bootstrap_sample(X,y)
            tree.fit(X_bt,y_bt)
            self.trees.append(tree)
            
    def _bootstrap_sample(self,X,y):
        n_samples=X.shape[0]
        indices=np.random.choice(n_samples,size=n_samples,replace=True)
        return X[indices], y[indices]
    
    def predict(self,x):
        predictions=np.array([tree.predict(x) for tree in self.trees])
        majority_votes=[np.bincount(pred).argmax() for pred in predictions.T]
        return np.array(majority_votes)
#### Analytics ####
def random_forest_tuning(X_train, y_train, X_val, y_val, d):
    T_values = [5, 10, 30, 50]
    max_features_values = [int(np.sqrt(d)), int(d/2)]
    best_acc = -1
    best_params = None
    results = []
    print("\n===== Random Forest Validation Results =====")
    print(f"{'T':<10} | {'max_features':<15} | {'Val Accuracy':<10}")
    print("-" * 45)
    for T in T_values:
        for mf in max_features_values:
            rf = RandomForest(
                n_trees=T,
                n_features=mf,
                max_depth=4,            
                min_sample_split=2  
            )		
            rf.fit(X_train, y_train)
            preds = rf.predict(X_val)
            acc = accuracy_score(y_val, preds)

            print(f"{T:<10} | {mf:<15} | {acc:.4f}")

            results.append((T, mf, acc))

            if acc > best_acc:
                best_acc = acc
                best_params = (T, mf)

    print("\nBEST RANDOM FOREST PARAMETERS:")
    print(f"T = {best_params[0]}, max_features = {best_params[1]}")
    print(f"Validation Accuracy = {best_acc:.4f}")

    return best_params, results

def run_random_forest():
    d = X_train.shape[1]   # number of features

    # Step 1 — tune
    best_params, _ = random_forest_tuning(X_train, y_train, X_val, y_val, d)
    best_T, best_mf = best_params
    # Step 2 — combine train and val sets
    X_train_val, y_train_val = dt.train_val_combine(X_train, y_train, X_val, y_val)

    # train final model
    final_rf = RandomForest(
        n_trees=best_T,
        n_features=best_mf,
        max_depth=4,
        min_sample_split=2
    )
    print("\n==== TRAINING FINAL RANDOM FOREST MODEL ====")
    final_rf.fit(X_train_val, y_train_val)

    # test
    
    test_preds = final_rf.predict(X_test)

    print("\n==== FINAL TEST PERFORMANCE ====")
    dt.evaluate_performance(y_test, test_preds)
run_random_forest()

