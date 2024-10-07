# Credit-Scoring-Model

## Project Overview
This project focuses on building a credit scoring model using a transaction dataset. The goal is to classify users into high-risk and low-risk groups based on transaction data and behavior patterns using RFMS formalism (Recency, Frequency, Monetary, Stability). The tasks include feature engineering, model training, evaluation, and comparison.



## Task Breakdown

### Task 2: Feature Engineering
Feature engineering was performed to generate new features and preprocess the data. This includes:
- **Aggregate Features**:
  - Total Transaction Amount per customer.
  - Average Transaction Amount per customer.
  - Transaction Count per customer.
  - Standard Deviation of Transaction Amounts per customer.
  
- **Extracted Features**:
  - Transaction Hour, Day, Month, Year.
  
- **Categorical Encoding**:
  - Categorical features such as `ProviderId`, `ProductCategory`, and `ChannelId` were encoded using **One-Hot Encoding**.

- **Handling Missing Values**:
  - Missing values were handled using imputation (mean/median/mode).

- **Normalization/Standardization**:
  - Numerical features were normalized to a [0, 1] range or standardized to have a mean of 0 and a standard deviation of 1.

### Task 3: Default Estimator and WoE Binning
- Users were classified into high-risk or low-risk categories using RFMS (Recency, Frequency, Monetary, Stability) formalism.
- **Weight of Evidence (WoE) Binning** was applied to enhance model interpretability and improve performance for credit scoring.

### Task 4: Model Selection and Training
- Models trained include:
  - **Logistic Regression**
  - **Random Forest**
- **Hyperparameter tuning** was performed using Grid Search to improve model performance.
- Evaluation metrics include:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC

### Task 4


## How to Run the Project

### Prerequisites
- Python 3.11 or higher
- Install the required packages by running:
  ```bash
  pip install -r requirements.txt
