import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

def extract_features(path):
    y, sr = librosa.load(path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

X, y = [], []
dataset_path = 'music_data/'

for label in os.listdir(dataset_path):
    label_dir = os.path.join(dataset_path, label)
    for file in os.listdir(label_dir):
        filepath = os.path.join(label_dir, file)
        features = extract_features(filepath)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'hidden_layer_sizes': [(64, 32)],
    'activation': ['relu'],
}

grid = GridSearchCV(MLPClassifier(max_iter=500), param_grid, cv=3)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
print("Best accuracy on test data:", grid.score(X_test, y_test))
print("Best parameters:", grid.best_params_)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(grid.best_estimator_, 'mlp_model.pkl')
