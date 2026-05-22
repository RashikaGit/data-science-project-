# ============================================================================
# SALES PREDICTION - SIMPLIFIED VERSION
# ============================================================================

print("="*60)
print("SALES PREDICTION PROJECT STARTING...")
print("="*60)

# Step 1: Libraries Import
print("\n[1/6] Importing libraries...")
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    print("✓ All libraries imported successfully!")
except Exception as e:
    print(f"✗ Error importing libraries: {e}")
    print("HINT: Run this command: pip install pandas numpy matplotlib seaborn scikit-learn")
    exit()

# Step 2: Create Sample Data
print("\n[2/6] Creating dataset...")
np.random.seed(42)
n = 200

df = pd.DataFrame({
    'TV_Advertising': np.random.uniform(100, 10000, n),
    'Radio_Advertising': np.random.uniform(50, 5000, n),
    'Social_Media_Advertising': np.random.uniform(100, 8000, n),
    'Print_Advertising': np.random.uniform(50, 3000, n),
    'Discount_Percentage': np.random.uniform(0, 30, n),
})

# Sales calculate karein
df['Sales'] = (
    50 + 
    df['TV_Advertising'] * 0.05 + 
    df['Radio_Advertising'] * 0.03 + 
    df['Social_Media_Advertising'] * 0.06 + 
    df['Print_Advertising'] * 0.02 +
    df['Discount_Percentage'] * 2 +
    np.random.normal(0, 50, n)
)

print(f"✓ Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nFirst 3 rows:\n{df.head(3)}")

# Step 3: Data Analysis
print("\n[3/6] Analyzing data...")
print(f"\nBasic Statistics:")
print(df.describe())

# Step 4: Prepare Data for Model
print("\n[4/6] Preparing data for modeling...")
X = df[['TV_Advertising', 'Radio_Advertising', 'Social_Media_Advertising', 
        'Print_Advertising', 'Discount_Percentage']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✓ Training data: {X_train.shape[0]} samples")
print(f"✓ Testing data: {X_test.shape[0]} samples")

# Step 5: Train Model
print("\n[5/6] Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("✓ Model trained successfully!")

# Step 6: Evaluate Model
print("\n[6/6] Evaluating model...")
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n{'='*60}")
print("RESULTS:")
print(f"{'='*60}")
print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")
print(f"RMSE: ${rmse:.2f}")
print(f"\nModel Coefficients:")
print(f"  Intercept: ${model.intercept_:.2f}")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")

# Prediction Example
print(f"\n{'='*60}")
print("SAMPLE PREDICTION:")
print(f"{'='*60}")
sample_input = [[5000, 2000, 4000, 1000, 10]]
predicted_sales = model.predict(sample_input)[0]
print(f"Input: TV=$5000, Radio=$2000, Social Media=$4000, Print=$1000, Discount=10%")
print(f"Predicted Sales: ${predicted_sales:,.2f}")

print(f"\n{'='*60}")
print("✅ PROJECT COMPLETED SUCCESSFULLY!")
print(f"{'='*60}")

# Show plot
print("\nGenerating visualization...")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Sales', fontsize=12)
plt.ylabel('Predicted Sales', fontsize=12)
plt.title('Actual vs Predicted Sales', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n🎉 All done!")