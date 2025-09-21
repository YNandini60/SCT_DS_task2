# Titanic Data Science Task 2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/hydra/Downloads/titanic.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())
print("\nSummary Statistics:\n", df.describe(include="all"))

print("\nMissing Values:\n", df.isnull().sum())

if "Age" in df.columns:
    df["Age"].fillna(df["Age"].median(), inplace=True)

if "Embarked" in df.columns:
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

if "Cabin" in df.columns:
    df.drop(columns=["Cabin"], inplace=True)

print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# 1. Survival Count
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Survived", palette="Set2")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# 2. Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Sex", hue="Survived", palette="pastel")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# 3. Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(data=df, x="Age", bins=20, kde=True, color="skyblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# 4. Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Pclass", hue="Survived", palette="muted")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
