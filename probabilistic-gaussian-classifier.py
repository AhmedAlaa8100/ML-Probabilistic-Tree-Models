import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sklearn.datasets
import matplotlib.pyplot as plt

# ================================================================
#  Part A1: Load, Split, Standardize
# ================================================================
digits = sklearn.datasets.load_digits()
X = digits.data
y = digits.target

# Train 70%, Validation 15%, Test 15%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)


# ================================================================
#  Part A2: Gaussian Generative Classifier Implementation
# ================================================================
class GaussianGenerativeClassifier:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.fitted = False

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape

        classes = np.unique(y)
        K = len(classes)

        pi = np.zeros(K)
        mu = np.zeros((K, d))

        # Priors and means
        for idx, k in enumerate(classes):
            Xk = X[y == k]
            pi[idx] = Xk.shape[0] / n
            mu[idx] = Xk.mean(axis=0)

        # Shared covariance
        Sigma = np.zeros((d, d))
        for idx, k in enumerate(classes):
            Xk = X[y == k]
            diffs = Xk - mu[idx]
            Sigma += diffs.T @ diffs
        Sigma /= n

        self.classes_ = classes
        self.pi_ = pi
        self.mu_ = mu
        self.Sigma_ = Sigma
        self.d_ = d
        self.K_ = K
        self.fitted = True
        return self

    def _prepare_cov(self, lam):
        d = self.d_
        Sigma_reg = self.Sigma_ + lam * np.eye(d)
        sign, logdet = np.linalg.slogdet(Sigma_reg)
        precision = np.linalg.inv(Sigma_reg)
        return precision, logdet

    def _log_scores(self, X, lam):
        X = np.asarray(X)
        n = X.shape[0]
        precision, logdet = self._prepare_cov(lam)

        const = -0.5 * (self.d_ * np.log(2 * np.pi) + logdet)

        scores = np.zeros((n, self.K_))
        for k in range(self.K_):
            diffs = X - self.mu_[k]
            quad = np.einsum('ij,ij->i', diffs @ precision, diffs)
            scores[:, k] = np.log(self.pi_[k]) + const - 0.5 * quad
        return scores

    def predict(self, X, lam):
        return self.classes_[np.argmax(self._log_scores(X, lam), axis=1)]

    def evaluate(self, X, y, lam):
        preds = self.predict(X, lam)
        return np.mean(preds == y)


# ================================================================
#  Part A3: Tune λ on Validation Set
# ================================================================
model = GaussianGenerativeClassifier()
model.fit(X_train, y_train)

lambda_candidates = [1e-4, 1e-3, 1e-2, 1e-1]
best_lam = None
best_acc = -1

for lam in lambda_candidates:
    acc = model.evaluate(X_val, y_val, lam)
    if acc > best_acc:
        best_acc = acc
        best_lam = lam


# ================================================================
#  Retrain on Train + Validation Using Best λ
# ================================================================
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.hstack([y_train, y_val])

final_model = GaussianGenerativeClassifier()
final_model.fit(X_train_full, y_train_full)


# ================================================================
#  Part A4: Final Test Set Evaluation
# ================================================================
test_preds = final_model.predict(X_test, best_lam)

test_accuracy  = accuracy_score(y_test, test_preds)
test_precision = precision_score(y_test, test_preds, average='macro')
test_recall    = recall_score(y_test, test_preds, average='macro')
test_f1        = f1_score(y_test, test_preds, average='macro')
test_cm        = confusion_matrix(y_test, test_preds)

print(f"\nBest λ: {best_lam:.4f}")

print("\n=== Test Set Performance ===")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")


# Plot confusion matrix
plt.figure(figsize=(7, 6))
plt.imshow(test_cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

# Tick labels 0–9
ticks = np.arange(10)
plt.xticks(ticks)
plt.yticks(ticks)

# Print numbers inside cells
for i in range(test_cm.shape[0]):
    for j in range(test_cm.shape[1]):
        plt.text(j, i, str(test_cm[i, j]),
                 ha='center', va='center')
plt.tight_layout()
plt.show()

