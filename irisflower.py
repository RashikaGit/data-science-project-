# ============================================================================
# IRIS FLOWER CLASSIFICATION - COMPLETE PROJECT
# ============================================================================

# Step 1: Libraries import karein
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Step 2: Dataset load karein
print("="*60)
print("LOADING DATASET...")
print("="*60)

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species_name'] = df['species'].map(species_map)

print(f"Dataset Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nSpecies Distribution:\n{df['species_name'].value_counts()}")

# Step 3: Data Visualization
print("\n" + "="*60)
print("CREATING VISUALIZATIONS...")
print("="*60)

# 3.1 Pairplot
sns.pairplot(df, hue='species_name', palette='viridis')
plt.suptitle('Iris Features Pairplot', y=1.02, fontsize=14, fontweight='bold')
plt.show()

# 3.2 Correlation Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
plt.show()

# Step 4: Data Preprocessing
print("\n" + "="*60)
print("PREPROCESSING DATA...")
print("="*60)

X = df[iris.feature_names]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Step 5: Train Multiple Models
print("\n" + "="*60)
print("TRAINING MODELS...")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'accuracy': accuracy, 'predictions': y_pred}
    print(f"{name:30s}: Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")

# Step 6: Find Best Model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']

print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print("="*60)

# Step 7: Visualize Results
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
colors = ['#2ecc71' if acc == best_accuracy else '#3498db' for acc in accuracies]

plt.bar(model_names, accuracies, color=colors, edgecolor='black')
plt.xlabel('Models', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('Model Comparison', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim([0.85, 1.0])
plt.show()

# Step 8: Classification Report for Best Model
print(f"\n{'='*60}")
print(f"CLASSIFICATION REPORT - {best_model_name}")
print('='*60)
y_pred_best = results[best_model_name]['predictions']
print(classification_report(y_test, y_pred_best, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Step 9: Make Predictions on New Data
print("\n" + "="*60)
print("TESTING NEW SAMPLES...")
print("="*60)

new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.9, 4.3, 1.3],
    [7.3, 3.0, 6.3, 1.8]
])

new_samples_scaled = scaler.transform(new_samples)
new_predictions = best_model.predict(new_samples_scaled)

for i, (sample, pred) in enumerate(zip(new_samples, new_predictions), 1):
    print(f"\nSample {i}: {sample}")
    print(f"Predicted Species: {iris.target_names[pred].upper()}")

# Final Summary
print("\n" + "="*70)
print(" " * 25 + "PROJECT COMPLETED!")
print("="*70)
print(f"""
✅ All tasks completed successfully!
✅ Best Model: {best_model_name}
✅ Accuracy: {best_accuracy*100:.2f}%
✅ Dataset: Iris ({df.shape[0]} samples, {len(iris.feature_names)} features)
✅ Classes: setosa, versicolor, virginica
""")
print("="*70)
print("Program Successfully running")