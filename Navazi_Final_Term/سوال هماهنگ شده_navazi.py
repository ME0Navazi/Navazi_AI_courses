import numpy as np
def gaussian(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
def anfis_forward(x1, x2, params):
    mu_A = [gaussian(x1, params["c_A1"], params["sigma_A1"]),
            gaussian(x1, params["c_A2"], params["sigma_A2"])]
    mu_B = [gaussian(x2, params["m_B1"], params["sigma_B1"]),
            gaussian(x2, params["m_B2"], params["sigma_B2"])]
    w = [
        mu_A[0] * mu_B[0],  # Rule 1: A1 & B1
        mu_A[0] * mu_B[1],  # Rule 2: A1 & B2
        mu_A[1] * mu_B[0],  # Rule 3: A2 & B1
        mu_A[1] * mu_B[1],  # Rule 4: A2 & B2
    ]
    S = sum(w)
    w_bar = [wi / S for wi in w]
    f = [
        params["a1"] * x1 + params["b1"] * x2 + params["c1"],
        params["a2"] * x1 + params["b2"] * x2 + params["c2"],
        params["a3"] * x1 + params["b3"] * x2 + params["c3"],
        params["a4"] * x1 + params["b4"] * x2 + params["c4"],
    ]
    y = sum(w_bar[i] * f[i] for i in range(4))
    return y, f, w, w_bar, S
def update_params(x1, x2, y_d, params, eta_a=0.01, eta_m=0.01):
    y, f, w, w_bar, S = anfis_forward(x1, x2, params)
    e = y_d - y
    grad_a2 = -e * w_bar[1] * x1
    params["a2"] -= eta_a * grad_a2  # gradient descent update
    grad_mB1 = 0.0
    for i in [0, 2]:  # rules that use B1
        grad_mB1 += (-e) * (f[i] - y) / S * w[i] * (x2 - params["m_B1"]) / (params["sigma_B1"] ** 2)
    params["m_B1"] -= eta_m * grad_mB1
    return params, y, e
params = {
    "c_A1": 0.0, "sigma_A1": 1.0,
    "c_A2": 1.0, "sigma_A2": 1.0,
    "m_B1": 0.0, "sigma_B1": 1.0,
    "m_B2": 1.0, "sigma_B2": 1.0,
    "a1": 0.5, "b1": 0.2, "c1": 0.1,
    "a2": -0.3, "b2": 0.4, "c2": 0.0,
    "a3": 0.1, "b3": -0.2, "c3": 0.3,
    "a4": 0.7, "b4": 0.5, "c4": -0.1,
}
x1, x2, y_d = 0.8, 0.5, 1.2
updated_params, y_pred, error = update_params(x1, x2, y_d, params)
print("Updated a2:", updated_params["a2"])
print("Updated m_B1:", updated_params["m_B1"])
print("Prediction:", y_pred, "Error:", error)
