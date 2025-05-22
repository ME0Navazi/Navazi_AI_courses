import random
import math

# تابع فعال‌سازی tansig و مشتق آن
def tansig(n):
    return 2 / (1 + math.exp(-2 * n)) - 1

def tansig_deriv(n):
    y = tansig(n)
    return 1 - y ** 2

# نرمال‌سازی مقیاسی داده‌ها به بازه [0,1]
def normalize(data):
    min_vals = [min(col) for col in zip(*data)]
    max_vals = [max(col) for col in zip(*data)]
    norm_data = []
    for row in data:
        norm_row = [(x - mn) / (mx - mn) if mx != mn else 0 for x, mn, mx in zip(row, min_vals, max_vals)]
        norm_data.append(norm_row)
    return norm_data

# مقداردهی اولیه شبکه عصبی
def initialize_network(layers):
    network = []
    for i in range(1, len(layers)):
        layer = []
        for _ in range(layers[i]):
            neuron = {
                'weights': [random.uniform(-0.6, 0.6) for _ in range(layers[i-1])],
                'bias': random.uniform(-1, 1)
            }
            layer.append(neuron)
        network.append(layer)
    return network

# پیش‌روی (Forward Propagation)
def forward_propagate(network, inputs):
    activations = [inputs]
    for layer in network:
        new_inputs = []
        for neuron in layer:
            z = sum(w * i for w, i in zip(neuron['weights'], inputs)) + neuron['bias']
            a = tansig(z)
            neuron['output'] = a
            neuron['z'] = z
            new_inputs.append(a)
        inputs = new_inputs
        activations.append(inputs)
    return activations

# پس‌انتشار خطا (Backpropagation)
def back_propagate(network, target, activations):
    deltas = []
    for l in reversed(range(len(network))):
        layer = network[l]
        errors = []
        if l == len(network) - 1:  # لایه خروجی
            for j, neuron in enumerate(layer):
                errors.append(target[j] - neuron['output'])
        else:  # لایه‌های مخفی
            for j in range(len(layer)):
                error = sum(next_layer[k]['weights'][j] * deltas[0][k] for k in range(len(next_layer)))
                errors.append(error)
        delta = [errors[j] * tansig_deriv(neuron['z']) for j, neuron in enumerate(layer)]
        deltas.insert(0, delta)
        next_layer = layer
    return deltas

# به‌روزرسانی وزن‌ها و بایاس‌ها
def update_weights(network, activations, deltas, lr):
    for l in range(len(network)):
        inputs = activations[l]
        for j, neuron in enumerate(network[l]):
            for k in range(len(inputs)):
                neuron['weights'][k] += lr * deltas[l][j] * inputs[k]
            neuron['bias'] += lr * deltas[l][j]

# آموزش شبکه عصبی
def train_mlp(network, train_inputs, train_targets, test_inputs, test_targets, lr=0.05, max_epochs=10000, tol=0.01):
    for epoch in range(max_epochs):
        train_errors = []
        for inputs, target in zip(train_inputs, train_targets):
            activations = forward_propagate(network, inputs)
            deltas = back_propagate(network, target, activations)
            update_weights(network, activations, deltas, lr)
            mse = sum((t - o) ** 2 for t, o in zip(target, activations[-1])) / len(target)
            train_errors.append(mse)

        test_errors = []
        for inputs, target in zip(test_inputs, test_targets):
            activations = forward_propagate(network, inputs)
            mse = sum((t - o) ** 2 for t, o in zip(target, activations[-1])) / len(target)
            test_errors.append(mse)

        avg_train = sum(train_errors) / len(train_errors)
        avg_test = sum(test_errors) / len(test_errors)

        if abs(avg_train - avg_test) < tol:
            print(f"\n✅ Training stopped at epoch {epoch + 1}")
            break
    return network

# تعریف معماری شبکه و داده‌ها
layers = [2, 3, 9, 2, 1]

train_inputs = [[0.1, 0.2], [0.8, 0.5], [0.3, 0.7], [0.6, 0.9]]
train_targets = [[0.1], [0.9], [0.4], [0.8]]

test_inputs = [[0.2, 0.3], [0.7, 0.6]]
test_targets = [[0.15], [0.85]]

# نرمال‌سازی
train_inputs = normalize(train_inputs)
test_inputs = normalize(test_inputs)

# آموزش
network = initialize_network(layers)
trained_net = train_mlp(network, train_inputs, train_targets, test_inputs, test_targets)

# نمایش نتایج تست
print("\n📊 Test Results:")
for inputs in test_inputs:
    output = forward_propagate(trained_net, inputs)[-1]
    print(f"🔹 Input: {inputs} → Output: {output}")

