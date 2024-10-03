import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def data_overview(df):
    """
    Print the overview of the dataset.
    """
    print("Dataset Overview:")
    print(df.info())

def summary_statistics(df):
    """
    Print summary statistics of the dataset.
    """
    print("\nSummary Statistics:")
    print(df.describe(include='all'))

def plot_numerical_distribution(df, numerical_columns):
    """
    Plot the distribution of numerical features.
    """
    df[numerical_columns].hist(figsize=(15, 10), bins=30, color='blue')
    plt.suptitle('Distribution of Numerical Features')
    plt.show()

def plot_categorical_distribution(df, categorical_columns):
    """
    Plot the distribution of categorical features.
    """
    for col in categorical_columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, data=df, palette='Set2')
        plt.title(f'Distribution of {col}')
        plt.show()

def correlation_analysis(df, numerical_columns):
    """
    Display a heatmap of correlations between numerical features.
    """
    correlation_matrix = df[numerical_columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()

def check_missing_values(df):
    """
    Print missing values in the dataset.
    """
    print("\nMissing Values:")
    print(df.isnull().sum())

def detect_outliers(df, numerical_columns):
    """
    Detect outliers using box plots.
    """
    for col in numerical_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot of {col}')
        plt.show()
