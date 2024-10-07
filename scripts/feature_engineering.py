import pandas as pd
import numpy as np
import category_encoders as ce
from xverse.transformer import WOE
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def load_data(filepath):
    """Load the CSV data."""
    return pd.read_csv(filepath)

def create_aggregate_features(data):
    """Create aggregate features for each customer."""
    data['Total_Transaction_Amount'] = data.groupby('CustomerId')['Amount'].transform('sum')
    data['Average_Transaction_Amount'] = data.groupby('CustomerId')['Amount'].transform('mean')
    data['Transaction_Count'] = data.groupby('CustomerId')['TransactionId'].transform('count')
    data['Std_Transaction_Amount'] = data.groupby('CustomerId')['Amount'].transform('std').fillna(0)
    return data

def extract_time_features(data):
    """Extract hour, day, month, and year from TransactionStartTime."""
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    data['Transaction_Hour'] = data['TransactionStartTime'].dt.hour
    data['Transaction_Day'] = data['TransactionStartTime'].dt.day
    data['Transaction_Month'] = data['TransactionStartTime'].dt.month
    data['Transaction_Year'] = data['TransactionStartTime'].dt.year
    return data

def encode_categorical_variables(data):
    """Encode categorical variables using one-hot and label encoding."""
    categorical_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    label_encoder = LabelEncoder()
    data['ProductId'] = label_encoder.fit_transform(data['ProductId'])
    return data

def handle_missing_values(data):
    """Handle missing values using median imputation."""
    imputer = SimpleImputer(strategy='median')
    data['Amount'] = imputer.fit_transform(data[['Amount']])
    data['Value'] = imputer.fit_transform(data[['Value']])
    return data

def normalize_and_standardize(data):
    """Normalize and standardize numerical features."""
    scaler = MinMaxScaler()
    data[['Amount', 'Value', 'Total_Transaction_Amount', 'Average_Transaction_Amount', 'Std_Transaction_Amount']] = scaler.fit_transform(
        data[['Amount', 'Value', 'Total_Transaction_Amount', 'Average_Transaction_Amount', 'Std_Transaction_Amount']])

    std_scaler = StandardScaler()
    data[['Transaction_Hour', 'Transaction_Day', 'Transaction_Month']] = std_scaler.fit_transform(
        data[['Transaction_Hour', 'Transaction_Day', 'Transaction_Month']])
    return data

def create_rfms_score(data):
    """Create RFMS score based on Recency, Frequency, Monetary, and Size."""
    data['Recency'] = data.groupby('CustomerId')['TransactionStartTime'].transform(lambda x: (x.max() - x).dt.days)
    data['Frequency'] = data.groupby('CustomerId')['TransactionId'].transform('count')
    data['Monetary'] = data.groupby('CustomerId')['Amount'].transform('sum')
    data['Size'] = data.groupby('CustomerId')['Amount'].transform('mean')

    threshold = data['Monetary'].mean() + data['Recency'].mean()
    data['RFMS_Score'] = (data['Recency'] + data['Monetary']).apply(lambda x: 'Good' if x >= threshold else 'Bad')
    return data

def apply_woe_binning(data):
    """Perform Weight of Evidence (WoE) binning using category_encoders."""
    
    # Ensure the necessary columns are present
    if 'RFMS_Score' not in data.columns or 'FraudResult' not in data.columns:
        raise ValueError("RFMS_Score or FraudResult column missing.")
    
    # Initialize the WoE encoder
    woe_encoder = ce.WOEEncoder(cols=['RFMS_Score'])
    
    # Fit the WoE encoder on 'RFMS_Score' and 'FraudResult' and transform the data
    data['RFMS_WoE'] = woe_encoder.fit_transform(data['RFMS_Score'], data['FraudResult'])
    
    return data

def feature_engineering_pipeline(filepath):
    """Complete feature engineering pipeline."""
    data = load_data(filepath)
    data = create_aggregate_features(data)
    data = extract_time_features(data)
    data = encode_categorical_variables(data)
    data = handle_missing_values(data)
    data = normalize_and_standardize(data)
    data = create_rfms_score(data)
    data = apply_woe_binning(data)
    
    return data

if __name__ == "__main__":
    filepath = '../data/data.csv'
    data = feature_engineering_pipeline(filepath)
    print(data.head())
