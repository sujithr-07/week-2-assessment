import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


# First split: 80% train + validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split: 60% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("Training size:", len(X_train))
print("Validation size:", len(X_val))
print("Testing size:", len(X_test))

#Overfitting demonstration with Decision Tree Classifier
depths = [1, 3, 5, 10, None]

train_scores = []
val_scores = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Training accuracy
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    # Validation accuracy
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    train_scores.append(train_acc)
    val_scores.append(val_acc)

    print(
        f"Depth {depth}: "
        f"Train {train_acc:.2f} "
        f"Val {val_acc:.2f}"
    )


#Plot vs Figure
plt.figure(figsize=(8, 5))

plt.plot(
    [str(d) for d in depths],
    train_scores,
    marker='o',
    label='Training Accuracy'
)

plt.plot(
    [str(d) for d in depths],
    val_scores,
    marker='o',
    label='Validation Accuracy'
)

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Overfitting Demonstration")
plt.legend()
plt.grid(True)

plt.show()

#Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

lr = LogisticRegression(max_iter=5000)

scores = cross_val_score(
    lr, X_train_scaled, y_train,
    cv=5, scoring='accuracy'
)

print(f"CV: {scores.mean():.2f} +/- {scores.std():.2f}")


#Data leakage demonstration

scaler = StandardScaler()

# Incorrect: fitting scaler on entire dataset
X_scaled = scaler.fit_transform(X)

# Split after scaling
X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Incorrect scaling completed")

# Fit scaler only on training data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# Transform validation and test data
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Correct scaling completed")