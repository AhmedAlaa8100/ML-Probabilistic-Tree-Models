import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load Dataset and Split
digits = sklearn.datasets.load_digits()
X, y = digits.data, digits.target

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Gaussian Generative Classifier
class GaussianGenerativeClassifier:

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.K_ = len(self.classes_)
        self.d_ = X.shape[1]

        self.pi_ = np.zeros(self.K_)
        self.mu_ = np.zeros((self.K_, self.d_))
        self.Sigma_ = np.zeros((self.d_, self.d_))

        for idx, k in enumerate(self.classes_):
            Xk = X[y == k]
            self.pi_[idx] = len(Xk) / len(X)
            self.mu_[idx] = Xk.mean(axis=0)
            diffs = Xk - self.mu_[idx]
            self.Sigma_ += diffs.T @ diffs

        self.Sigma_ /= len(X)

    def _prepare_cov(self, lam):
        Sigma_reg = self.Sigma_ + lam * np.eye(self.d_)
        sign, logdet = np.linalg.slogdet(Sigma_reg)
        precision = np.linalg.inv(Sigma_reg)
        return precision, logdet

    def _log_scores(self, X, lam):
        precision, logdet = self._prepare_cov(lam)
        const = -0.5 * (self.d_ * np.log(2 * np.pi) + logdet)

        scores = np.zeros((X.shape[0], self.K_))
        for k in range(self.K_):
            diffs = X - self.mu_[k]
            quad = np.einsum("ij,ij->i", diffs @ precision, diffs)
            scores[:, k] = np.log(self.pi_[k]) + const - 0.5 * quad
        return scores

    def predict(self, X, lam):
        scores = self._log_scores(X, lam)
        return self.classes_[np.argmax(scores, axis=1)]

    def evaluate(self, X, y, lam):
        preds = self.predict(X, lam)
        return np.mean(preds == y)

# Hyperparameter Tuning
model = GaussianGenerativeClassifier()
model.fit(X_train, y_train)

lambda_candidates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
best_acc = -1
best_lam = None

print("=== Hyperparameter Tuning ===")
for lam in lambda_candidates:
    acc = model.evaluate(X_val, y_val, lam)
    print(f"λ = {lam:.6f} → Validation Accuracy = {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_lam = lam

print(f"\nBest λ Selected: {best_lam:.6f}")

# Retrain on Train + Validation
X_full = np.vstack([X_train, X_val])
y_full = np.hstack([y_train, y_val])

final_model = GaussianGenerativeClassifier()
final_model.fit(X_full, y_full)

# Final Test Evaluation
test_preds = final_model.predict(X_test, best_lam)

test_acc  = accuracy_score(y_test, test_preds)
test_prec = precision_score(y_test, test_preds, average="macro")
test_rec  = recall_score(y_test, test_preds, average="macro")
test_f1   = f1_score(y_test, test_preds, average="macro")

print("\n=== Final Test Metrics ===")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall:    {test_rec:.4f}")
print(f"F1-Score:  {test_f1:.4f}")

# Confusion Matrix Plot
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap='Blues')
plt.title(f"Confusion Matrix (λ = {best_lam})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

ticks = np.arange(10)
plt.xticks(ticks)
plt.yticks(ticks)

for i in range(10):
    for j in range(10):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color=color)

plt.tight_layout()
plt.show()
