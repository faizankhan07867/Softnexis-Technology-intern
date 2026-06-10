# PROJECT 1: Exploratory Data Analysis & Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df=pd.read_csv(url)

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

print("\nMissing Values:\n", df.isnull().sum())

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nSurvival Rate:\n", df['Survived'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Survived')
plt.title('Survival Count')
plt.savefig('survival_count.png')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.savefig('age_distribution.png')
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived')
plt.title('Age vs Fare')
plt.savefig('age_fare_scatter.png')
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

print("EDA Project Completed Successfully!")
