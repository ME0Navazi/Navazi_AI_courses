
!pip install --quiet numpy pandas scikit-learn matplotlib

# 1) Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
np.random.seed(42)

# 2) Load dataset if user uploaded solar_dataset.csv to /content, otherwise generate it
csv_path_colab = "/content/solar_dataset.csv"
if os.path.exists(csv_path_colab):
    df = pd.read_csv(csv_path_colab)
    print("Loaded dataset from /content/solar_dataset.csv")
else:
    print("No CSV found in /content — generating synthetic dataset (same distribution as provided).")
    N = 2000
    G = np.random.uniform(200, 1000, N)        # Irradiance W/m^2
    T = np.random.uniform(0, 40, N)            # Temperature C
    theta = np.random.uniform(0, 90, N)        # Incidence angle degrees
    eta0 = 0.20
    alpha = 0.004
    A = 1.6
    theta_rad = np.deg2rad(theta)
    P_clean = eta0 * A * G * np.cos(theta_rad) * (1 - alpha * (T - 25))
    noise = np.random.normal(0, 5.0, N)
    P = P_clean + noise
    df = pd.DataFrame({
        "G_Wm2": G,
        "T_C": T,
        "theta_deg": theta,
        "P_W": P,
        "P_clean_W": P_clean,
        "noise_W": noise
    })
    # optional: save
    df.to_csv(csv_path_colab, index=False)
    print(f"Generated and saved dataset to {csv_path_colab}")

print(df.head().round(4))
print(df[["G_Wm2","T_C","theta_deg","P_W"]].describe().round(3))

# 3) Preprocessing: inputs and normalization
X = df[["G_Wm2","T_C","theta_deg"]].values
y = df[["P_W"]].values

scaler_X = MinMaxScaler()
X_norm = scaler_X.fit_transform(X)   # maps inputs to [0,1] for stable MF init
# Optionally scale y if desired (we'll keep y in original units to interpret MSE in watts)

# 4) ANFIS implementation (simple, numeric gradient for premise)
class SimpleANFIS:
    def __init__(self, n_inputs=3, n_mfs=2):
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.n_rules = n_mfs ** n_inputs
        # initialize premise params (centers in [0.2,0.8], sigmas reasonable)
        centers = np.linspace(0.2, 0.8, n_mfs)
        self.centers = np.array([centers.copy() for _ in range(n_inputs)])
        self.sigmas = np.ones_like(self.centers) * 0.18
        # consequents: each rule has linear params (p0 + p1*x1 + ... + pn*xn)
        self.n_conseq_params = 1 + n_inputs
        self.consequents = np.zeros((self.n_rules, self.n_conseq_params))
        grids = np.array(np.meshgrid(*[np.arange(n_mfs) for _ in range(n_inputs)], indexing='ij'))
        self.rules_mf_indices = grids.reshape(n_inputs, -1).T

    def gaussian(self, x, c, s):
        return np.exp(-0.5 * ((x - c)/s)**2)

    def rule_firing_strengths(self, X):
        N = X.shape[0]
        # membership values (n_inputs, n_mfs, N)
        mvals = np.zeros((self.n_inputs, self.n_mfs, N))
        for i in range(self.n_inputs):
            for j in range(self.n_mfs):
                mvals[i,j,:] = self.gaussian(X[:,i], self.centers[i,j], self.sigmas[i,j])
        w = np.ones((N, self.n_rules))
        for r in range(self.n_rules):
            for i in range(self.n_inputs):
                mfidx = self.rules_mf_indices[r, i]
                w[:, r] *= mvals[i, mfidx, :]
        return w

    def predict(self, X):
        w = self.rule_firing_strengths(X)
        w_sum = np.sum(w, axis=1, keepdims=True)
        w_sum[w_sum == 0] = 1e-10
        w_norm = w / w_sum
        X_design = np.hstack([np.ones((X.shape[0],1)), X])
        rule_outputs = X_design.dot(self.consequents.T)
        y_pred = np.sum(w_norm * rule_outputs, axis=1, keepdims=True)
        return y_pred, w, w_norm, rule_outputs

    def update_consequents_by_least_squares(self, X, y):
        N = X.shape[0]
        w = self.rule_firing_strengths(X)
        w_sum = np.sum(w, axis=1, keepdims=True); w_sum[w_sum==0]=1e-10
        w_norm = w / w_sum
        X_design = np.hstack([np.ones((N,1)), X])
        Phi = np.zeros((N, self.n_rules * self.n_conseq_params))
        for r in range(self.n_rules):
            Phi[:, r*self.n_conseq_params:(r+1)*self.n_conseq_params] = w_norm[:, r:r+1] * X_design
        theta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        self.consequents = theta.reshape(self.n_rules, self.n_conseq_params)

    def loss(self, X, y):
        y_pred, *_ = self.predict(X)
        return np.mean((y - y_pred)**2)

    def numeric_grad_premise(self, X_batch, y_batch, eps=1e-4):
        grad_centers = np.zeros_like(self.centers)
        grad_sigmas = np.zeros_like(self.sigmas)
        base_loss = self.loss(X_batch, y_batch)
        for i in range(self.n_inputs):
            for j in range(self.n_mfs):
                orig = self.centers[i,j]
                self.centers[i,j] = orig + eps
                self.update_consequents_by_least_squares(X_batch, y_batch)
                plus_loss = self.loss(X_batch, y_batch)
                self.centers[i,j] = orig
                grad_centers[i,j] = (plus_loss - base_loss) / eps
        for i in range(self.n_inputs):
            for j in range(self.n_mfs):
                orig = self.sigmas[i,j]
                self.sigmas[i,j] = orig + eps
                self.update_consequents_by_least_squares(X_batch, y_batch)
                plus_loss = self.loss(X_batch, y_batch)
                self.sigmas[i,j] = orig
                grad_sigmas[i,j] = (plus_loss - base_loss) / eps
        self.update_consequents_by_least_squares(X_batch, y_batch)
        return grad_centers, grad_sigmas

    def train(self, X, y, X_val=None, y_val=None, epochs=30, lr_premise=0.05, batch_for_grad=400):
        history = {"train_loss": [], "val_loss": []}
        N = X.shape[0]
        self.update_consequents_by_least_squares(X, y)
        for ep in range(epochs):
            self.update_consequents_by_least_squares(X, y)
            train_loss = self.loss(X, y)
            val_loss = self.loss(X_val, y_val) if (X_val is not None) else None
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss if val_loss is not None else np.nan)
            idx = np.random.choice(N, min(batch_for_grad, N), replace=False)
            X_batch = X[idx]; y_batch = y[idx]
            grad_c, grad_s = self.numeric_grad_premise(X_batch, y_batch, eps=1e-4)
            self.centers -= lr_premise * grad_c
            self.sigmas -= lr_premise * grad_s
            self.sigmas = np.clip(self.sigmas, 1e-3, 2.0)
            self.update_consequents_by_least_squares(X, y)
            if (ep+1) % 5 == 0 or ep == 0:
                if val_loss is not None:
                    print(f"Epoch {ep+1}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
                else:
                    print(f"Epoch {ep+1}/{epochs}  train_loss={train_loss:.6f}")
        return history

# 5) Split data
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=1)

# 6) Train ANFIS
anfis = SimpleANFIS(n_inputs=3, n_mfs=2)
history = anfis.train(X_train, y_train, X_val=X_test, y_val=y_test,
                      epochs=30, lr_premise=0.05, batch_for_grad=400)

# 7) Evaluate
y_train_pred, *_ = anfis.predict(X_train)
y_test_pred, *_ = anfis.predict(X_test)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"Final Train MSE: {mse_train:.6f}   Test MSE: {mse_test:.6f}")

# 8) Plots
plt.figure(figsize=(6,3.5))
plt.plot(history["train_loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training history")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_test_pred, s=10)
plt.xlabel("True P (W)")
plt.ylabel("Predicted P (W)")
plt.title("True vs Predicted (test)")
plt.tight_layout()
plt.show()

# 9) Save learned parameters for later use
params = {
    "scaler_X": scaler_X,
    "centers": anfis.centers,
    "sigmas": anfis.sigmas,
    "consequents": anfis.consequents
}
with open("/content/anfis_params.pkl", "wb") as f:
    pickle.dump(params, f)
print("Saved parameters to /content/anfis_params.pkl")
