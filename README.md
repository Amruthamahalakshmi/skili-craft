import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic styles for plots
sns.set_style('whitegrid')

# Load the dataset
# Assuming the file 'train.csv' is in the same directory
try:
    titanic_df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Please make sure 'train.csv' is in the same directory.")
    exit()

# --- 1. Data Cleaning ---

print("--- Initial Data Snapshot ---")
print(titanic_df.head())
print("\n--- Data Information ---")
titanic_df.info()
print("\n--- Missing Values ---")
print(titanic_df.isnull().sum())

# Handling missing values
# 1. Age: Impute missing ages using the median age for each Title
titanic_df['Title'] = titanic_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
common_titles = ['Mr', 'Mrs', 'Miss', 'Master', 'Dr']
titanic_df['Title'] = titanic_df['Title'].apply(lambda x: x if x in common_titles else 'Other')

# Map titles to a numerical format for convenience
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Other": 6}
titanic_df['Title'] = titanic_df['Title'].map(title_mapping)

for title in titanic_df['Title'].unique():
    median_age = titanic_df[titanic_df['Title'] == title]['Age'].median()
    titanic_df.loc[(titanic_df['Age'].isnull()) & (titanic_df['Title'] == title), 'Age'] = median_age

# 2. Embarked: Impute with the most common port
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

# 3. Cabin: This column has too many missing values. We can create a new feature
#    'Has_Cabin' to indicate if a passenger had a cabin recorded.
titanic_df['Has_Cabin'] = titanic_df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
# The original Cabin column can be dropped
titanic_df.drop('Cabin', axis=1, inplace=True)

# 4. Fare: One missing value, impute with the median fare
titanic_df['Fare'].fillna(titanic_df['Fare'].median(), inplace=True)

# Feature Engineering
# 1. Family Size
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1

# 2. IsAlone
titanic_df['IsAlone'] = 0
titanic_df.loc[titanic_df['FamilySize'] == 1, 'IsAlone'] = 1

# Drop columns that are not needed or have been transformed
titanic_df.drop(['Name', 'Ticket', 'SibSp', 'Parch'], axis=1, inplace=True)

# Convert categorical features to numerical
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
titanic_df['Embarked'] = titanic_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# --- 2. Exploratory Data Analysis (EDA) ---

print("\n--- Cleaned Data Information ---")
titanic_df.info()
print("\n--- Cleaned Data Head ---")
print(titanic_df.head())
print("\n--- Missing Values after Cleaning ---")
print(titanic_df.isnull().sum())

# Survival Rate
survival_rate = titanic_df['Survived'].value_counts(normalize=True) * 100
print(f"\nOverall Survival Rate: {survival_rate[1]:.2f}%")

# Plotting survival by key features
def plot_survival_by(feature, title, x_label):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature, y='Survived', data=titanic_df, palette='viridis')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Survival Rate')
    plt.show()

# 1. Survival by Sex
plot_survival_by('Sex', 'Survival Rate by Gender (0=Male, 1=Female)', 'Gender')

# 2. Survival by Passenger Class
plot_survival_by('Pclass', 'Survival Rate by Passenger Class', 'Passenger Class')

# 3. Survival by Embarkation Port
plot_survival_by('Embarked', 'Survival Rate by Embarkation Port (0=S, 1=C, 2=Q)', 'Port of Embarkation')

# 4. Survival by Family Size
plot_survival_by('FamilySize', 'Survival Rate by Family Size', 'Family Size')

# 5. Survival by Age
plt.figure(figsize=(12, 7))
sns.kdeplot(titanic_df[titanic_df['Survived'] == 1]['Age'], label='Survived', shade=True, color='skyblue')
sns.kdeplot(titanic_df[titanic_df['Survived'] == 0]['Age'], label='Did Not Survive', shade=True, color='salmon')
plt.title('Age Distribution for Survivors vs. Non-Survivors')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.show()

# 6. Survival by Fare
plt.figure(figsize=(12, 7))
sns.kdeplot(titanic_df[titanic_df['Survived'] == 1]['Fare'], label='Survived', shade=True, color='skyblue')
sns.kdeplot(titanic_df[titanic_df['Survived'] == 0]['Fare'], label='Did Not Survive', shade=True, color='salmon')
plt.title('Fare Distribution for Survivors vs. Non-Survivors')
plt.xlabel('Fare')
plt.ylabel('Density')
plt.legend()
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
correlation_matrix = titanic_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Variables')
plt.show()

print("\n--- Key Findings Summary ---")
print("1. Gender (Sex) is the most significant predictor of survival. Females had a much higher survival rate.")
print("2. Passenger Class (Pclass) strongly correlates with survival. First-class passengers had a significantly better chance of surviving.")
print("3. Age plays a role, with children and younger individuals having a better chance of survival, a trend more pronounced among males.")
print("4. Fare is positively correlated with survival, which is a proxy for Pclass and social status.")
print("5. Traveling alone seems to have a negative impact on survival compared to having a small family.")
