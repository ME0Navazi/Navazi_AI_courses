import numpy as np

# ---------- Utilities ----------
def gaussian(x, center, sigma):
    """Gaussian MF"""
    return np.exp(-((x - center) ** 2) / (2.0 * (sigma ** 2) + 1e-12))

# ---------- ANFIS (Sugeno) forward pass ----------
def anfis_forward(x1, x2, params, eps=1e-12):
    # Stage 1: fuzzification
    mu_A1 = gaussian(x1, params["c_A1"], params["sigma_A1"])
    mu_A2 = gaussian(x1, params["c_A2"], params["sigma_A2"])
    mu_B1 = gaussian(x2, params["m_B1"], params["sigma_B1"])
    mu_B2 = gaussian(x2, params["m_B2"], params["sigma_B2"])

    # Stage 2: rule firing strengths
    # Ordering: [A1B1, A1B2, A2B1, A2B2]
    w = np.array([
        mu_A1 * mu_B1,  # rule 1
        mu_A1 * mu_B2,  # rule 2
        mu_A2 * mu_B1,  # rule 3
        mu_A2 * mu_B2   # rule 4
    ])

    # Stage 3: aggregation (for Sugeno, we compute normalized weights)
    S = np.sum(w) + eps
    w_bar = w / S

    # Consequents (first-order Sugeno)
    f = np.array([
        params["a1"] * x1 + params["b1"] * x2 + params["c1"],
        params["a2"] * x1 + params["b2"] * x2 + params["c2"],
        params["a3"] * x1 + params["b3"] * x2 + params["c3"],
        params["a4"] * x1 + params["b4"] * x2 + params["c4"]
    ])

    # Stage 4: defuzzification (Sugeno weighted average)
    y = np.dot(w_bar, f)  # same as sum(w_i * f_i) / S

    # return many useful values for gradient calculation
    mu = {"A1": mu_A1, "A2": mu_A2, "B1": mu_B1, "B2": mu_B2}
    return {
        "y": float(y),
        "f": f,
        "w": w,
        "w_bar": w_bar,
        "S": float(S),
        "mu": mu
    }

# ---------- Gradient computations and one update ----------
def compute_and_update(x1, x2, y_d, params, eta_a=0.01, eta_m=0.01):
    out = anfis_forward(x1, x2, params)
    y = out["y"]
    f = out["f"]
    w = out["w"]
    w_bar = out["w_bar"]
    S = out["S"]
    mu = out["mu"]

    e = y_d - y
    J = 0.5 * e * e

    # Gradient wrt a2 (rule index 1 in zero-based indexing)
    grad_a2 = -e * w_bar[1] * x1   # ∂J/∂a2 = -e * bar_w2 * x1
    params["a2"] = params["a2"] - eta_a * grad_a2

    # Gradient wrt m_B1 (center of B1)
    # rules that use B1: indices 0 and 2 (A1B1 and A2B1)
    grad_mB1 = 0.0
    sigma_B1 = params["sigma_B1"]
    mu_B1 = mu["B1"]
    for i in [0, 2]:
        # ∂J/∂w_i = -e * (f_i - y) / S
        dJ_dw = -e * (f[i] - y) / S
        # ∂w_i/∂mu_B1 = mu_A_{p(i)}   (muA for rule i)
        if i == 0:
            muA = mu["A1"]
        else:
            muA = mu["A2"]
        # ∂mu_B1/∂m_B1 = mu_B1 * (x2 - m_B1) / sigma_B1^2
        dmu_dm = mu_B1 * (x2 - params["m_B1"]) / (sigma_B1 ** 2 + 1e-12)
        # multiply chain
        grad_mB1 += dJ_dw * muA * dmu_dm

    params["m_B1"] = params["m_B1"] - eta_m * grad_mB1

    return {
        "params": params,
        "y": y,
        "error": e,
        "J": J,
        "grad_a2": grad_a2,
        "grad_mB1": grad_mB1
    }

# ---------- Example usage ----------
if __name__ == "__main__":
    # Example initial parameters (you can change these)
    params = {
        # A MFs on x1
        "c_A1": 0.0, "sigma_A1": 1.0,
        "c_A2": 1.0, "sigma_A2": 1.0,
        # B MFs on x2
        "m_B1": 0.0, "sigma_B1": 1.0,
        "m_B2": 1.0, "sigma_B2": 1.0,
        # consequents (a_i, b_i, c_i) for 4 rules
        "a1": 0.5, "b1": 0.2, "c1": 0.1,
        "a2": -0.3, "b2": 0.4, "c2": 0.0,
        "a3": 0.1, "b3": -0.2, "c3": 0.3,
        "a4": 0.7, "b4": 0.5, "c4": -0.1
    }

    # single training sample
    x1, x2, y_d = 0.8, 0.5, 1.2
    result = compute_and_update(x1, x2, y_d, params, eta_a=0.01, eta_m=0.005)

    print("Prediction y before update (approx):", result["y"])
    print("Error e:", result["error"])
    print("Cost J:", result["J"])
    print("Gradient ∂J/∂a2:", result["grad_a2"])
    print("Gradient ∂J/∂m_B1:", result["grad_mB1"])
    print("Updated a2:", result["params"]["a2"])
    print("Updated m_B1:", result["params"]["m_B1"])
