
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

## Data class to hold the tree nodes
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        if self.value is not None:
            return True
        return False


## Model Implementation
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=2,max_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.root=None
        self.max_features=max_features
        self.all_gains=[]
        
    def fit(self, X, y):
        self.all_gains = [0] * X.shape[1]
        # recursively build the tree
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))    
        # stopping criteria
        if(depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        # Determine the best split
        best_feature, best_threshold = self._best_split(X, y)
        # mask data to given split
        left_mask=X[:, best_feature] <= best_threshold
        right_mask=X[:, best_feature] > best_threshold
        left_data, right_data = X[left_mask], X[right_mask]
        left_labels, right_labels = y[left_mask], y[right_mask]
        # recursively grow the left and right children each with different data
        left_child=self._grow_tree(left_data, left_labels, depth+1)
        right_child=self._grow_tree(right_data, right_labels, depth+1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_overall_gain = -1
        best_feature = None
        best_threshold = None
        if self.max_features is None or self.max_features > n_features:
            features_to_consider = range(n_features)
        else:
            features_to_consider = np.random.choice(n_features, self.max_features, replace=False)


        for feature in features_to_consider:

            # Compute all possible midpoints for this feature
            values = np.sort(np.unique(X[:, feature]))
            thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values) - 1)]

            best_feature_gain = -1  # reset best gain for this feature

            for threshold in thresholds:
                gain = self._info_gain(y, X[:, feature], threshold)

                # Update best-overall split
                if gain > best_overall_gain:
                    best_overall_gain = gain
                    best_feature = feature
                    best_threshold = threshold

                # Update best gain for THIS feature
                if gain > best_feature_gain:
                    best_feature_gain = gain

            # Save best gain of this feature for feature ranking
            self.all_gains[feature] = best_feature_gain

        return best_feature, best_threshold

    def _info_gain(self, y, feature_column, threshold):
        parent_entropy=self.entropy(y)
        left_mask=feature_column <= threshold
        right_mask=feature_column > threshold
        left_y=y[left_mask]
        right_y=y[right_mask]
        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0
        left_entropy=self.entropy(left_y)
        right_entropy=self.entropy(right_y)
        weighted_entropy=(len(left_y)/len(y))*left_entropy + (len(right_y)/len(y))*right_entropy
        info_gain=parent_entropy - weighted_entropy
        return info_gain
    
    def entropy(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        return entropy
    
    def _most_common_label(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        most_common = class_labels[np.argmax(counts)]
        return most_common
    
    def predict(self, X):
        root=self.root
        predictions=[]
        for x in X:
            node = self.root
            while not node.is_leaf_node():
                if x[node.feature] <= node.threshold:
                    node=node.left
                else:
                    node=node.right
            predictions.append(node.value)
        return np.array(predictions)
    
    def ranked_features(self):
       
        if not self.all_gains:
            return 0
        gains=np.argsort(self.all_gains)[::-1]
        table = pd.DataFrame({
        "Feature": [feature_names[i] for i in gains],
        "Information Gain": [self.all_gains[i] for i in gains]
        })
        print(table)
        
        
######## Analytics Functions ##########
def hyperparameter_tuning(X_train,y_train,X_val,y_val):
    best_acc=0
    best_params = None
    accuracy=[]
    best_tree=None
    max_depth={2, 4, 6, 8, 10} 
    min_samples_split = {2, 5, 10}
   
    results = []

    for depth in max_depth:
        for min_split in min_samples_split:
            tree = DecisionTree(max_depth=depth, min_samples_split=min_split)
            tree.fit(X_train, y_train)
            preds = tree.predict(X_val)
            acc = np.mean(preds == y_val)
            print(f"Depth={depth}, MinSplit={min_split}, ValAcc={acc:.4f}")
            results.append((depth, min_split, acc))
            if acc > best_acc:
                best_acc = acc
                best_params = (depth, min_split)
         
    print("\nBest Parameters:", best_params, "Validation Accuracy:", best_acc)
    return best_params, results
 
def max_depth_analysis(X_train,y_train,X_val,y_val):
    max_depth={2, 4, 6, 8, 10} 
    min_samples_split=2
    table = []
    for depth in max_depth:
        tree = DecisionTree(max_depth=depth, min_samples_split=min_samples_split)
        tree.fit(X_train, y_train)
        preds = tree.predict(X_val)
        val_acc = np.mean(preds == y_val)
        train_preds = tree.predict(X_train)
        train_acc = np.mean(train_preds == y_train)
        print(f"Depth={depth}, ValAcc={val_acc:.4f}")
        
        table.append((depth, train_acc, val_acc))

    print("\nDepth | Train Acc | Val Acc")
    for row in table:
        print(f"{row[0]:5} | {row[1]:9.4f} | {row[2]:8.4f}")

def train_val_combine(X_train, y_train, X_val, y_val):
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.hstack((y_train, y_val))
    return X_train_val, y_train_val

def evaluate_performance(y_true, y_pred):
    print("===== PERFORMANCE METRICS =====")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average=None)}")
    print(f"Recall:    {recall_score(y_true, y_pred, average=None)}")
    print(f"F1-score:  {f1_score(y_true, y_pred, average=None)}\n")

    print("===== CLASSIFICATION REPORT =====")
    print(classification_report(y_true, y_pred, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("===== CONFUSION MATRIX =====")
    print(cm)
   
def run_decision_tree():
    best_params, tuning_results = hyperparameter_tuning(X_train, y_train, X_val, y_val)
    best_depth, best_min_split = best_params
    final_tree = DecisionTree(max_depth=best_depth, min_samples_split=best_min_split)
    X_train_val, y_train_val = train_val_combine(X_train, y_train, X_val, y_val)
    max_depth_analysis(X_train, y_train, X_val, y_val)
    final_tree.fit(X_train_val, y_train_val)
    test_preds = final_tree.predict(X_test)
    evaluate_performance(y_test, test_preds)
    final_tree.ranked_features()
     

    