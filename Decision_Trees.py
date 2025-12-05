import numpy as np


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


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.root=None
        
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        if(depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        best_feature, best_threshold = self._best_split(X, y, n_features)
        left_mask=X[:, best_feature] <= best_threshold
        right_mask=X[:, best_feature] > best_threshold
        left_data, right_data = X[left_mask], X[right_mask]
        left_labels, right_labels = y[left_mask], y[right_mask]
        left_child=self._grow_tree(left_data, left_labels, depth+1)
        right_child=self._grow_tree(right_data, right_labels, depth+1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    def _best_split(self, X, y):
        n_samples,n_features=X.shape
        best_gain=-1
        split_ft, split_threshold=None, None
        for feature in range(n_features):
            values=sorted(np.unique(X[:, feature]))
            thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values) - 1)]
            for threshold in thresholds:
                gain=self._info_gain(y, X[:, feature], threshold)
                if gain>best_gain:
                    best_gain=gain
                    split_ft=feature
                    split_threshold=threshold
        return split_ft, split_threshold
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