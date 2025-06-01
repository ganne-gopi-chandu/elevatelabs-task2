import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("cleaned_data_final.csv")

# 1. Summary statistics
print("\n--- Summary Statistics ---")
print(df.describe(include='all'))  # Includes numeric and categorical

# 2. Histograms and Boxplots for numerical features
numeric_cols = list(df.select_dtypes(include=['float64', 'int64']).columns)  # Ensure list format

# Histograms
print("\n--- Histograms ---")
df[numeric_cols].hist(bins=20, figsize=(12, 8), edgecolor='black')
plt.tight_layout()
plt.show()

# Boxplots
print("\n--- Boxplots ---")
for col in numeric_cols:
    plt.figure(figsize=(6, 2))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()

# 3. Correlation matrix and pairplot
if len(numeric_cols) > 1:  # Ensure correlation calculation
    print("\n--- Correlation Matrix ---")
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    # Pairplot (for smaller datasets)
    print("\n--- Pairplot ---")
    sns.pairplot(df[numeric_cols], diag_kind='kde')
    plt.show()

# 4. Identify patterns/trends/anomalies (manual step via visuals)
print("\nLook at the boxplots and histograms for:")
print("- Skewed distributions")
print("- Outliers (boxplot whiskers)")
print("- Strong correlations (heatmap)")

# 5. Basic inferences
if 'Survived' in df.columns:
    print("\n--- Basic Inferences ---")
    if 'Sex' in df.columns:
        print("- Checking survival rate by gender:")
        print(df.groupby('Sex')['Survived'].mean())

    if 'Fare' in numeric_cols and 'Age' in numeric_cols:  # Ensure columns exist
        print("- Correlation with Fare, Age:")
        print(corr_matrix['Survived'].sort_values(ascending=False))

print("\nUse visual patterns to note:")
print("- Which features show outliers?")
print("- Are any features highly correlated?")
print("- Which features differ clearly by target class (e.g., Survived)?")

