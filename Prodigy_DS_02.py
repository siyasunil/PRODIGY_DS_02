from email.quoprimime import body_length

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import xlabel

# Load dataset
df = pd.read_csv("train.csv")

# Data info & missing values
print(df.info())
print(df.isnull().sum())

# Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])

# Map Survived column for readability
df['Survived_label'] = df['Survived'].map({0: 'Not Survived', 1: 'Survived'})

# Pie Chart: Survival Percentage
survival_counts = df['Survived_label'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(survival_counts, labels=survival_counts.index,
        colors=['#ff9999','#66b3ff'], autopct='%1.1f%%', startangle=90)
plt.title('Survival Percentage')
plt.show()

# Survival Rate by Gender
survival_rate_gender = df.groupby('Sex')['Survived'].mean().reset_index()
sns.barplot(x='Sex', y='Survived', data=survival_rate_gender, palette='dark', hue='Sex')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Gender')
plt.show()

# Survival Rate by Passenger Class (Pclass)
survival_rate_class = df.groupby('Pclass')['Survived'].mean().reset_index()
sns.barplot(x='Pclass', y='Survived', data=survival_rate_class, palette='bright', hue='Pclass')
plt.ylabel('Survival Rate')
xlabel('Passenger Class')
plt.title('Survival Rate by Passenger Class')
plt.show()

# Age Distribution by Survival Status (Boxplot)
sns.boxplot(x='Survived_label', y='Age', data=df, palette='pastel', hue='Survived')
xlabel('Survival')
plt.title('Age Distribution by Survival Status')
plt.show()

# Pairplot of key features colored by survival
sns.pairplot(df[['Survived_label','Age','Fare','Pclass']], hue='Survived_label', palette='husl')
plt.show()

# Correlation Heatmap (only numeric columns)
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(8,12))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
