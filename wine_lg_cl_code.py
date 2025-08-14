from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Load and explore the data:

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="class")
df = pd.concat([X, y], axis=1)

print(df.head())
print(df.describe())
print(df.info())
print(df['class'].value_counts())
print(df.isna().sum())

# Split the data:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model:

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions:

y_pred = model.predict(X_test)

# Evaluate the accuracy:

acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2%}")

