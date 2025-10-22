import pandas as pd

def load_data():
    # Download German Credit dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    columns = ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
           'savings_status', 'employment', 'installment_rate', 'personal_status', 
           'other_parties', 'residence_since', 'property_magnitude', 'age', 
           'other_payment_plans', 'housing', 'existing_credits', 'job', 
           'num_dependents', 'own_telephone', 'foreign_worker', 'target']

    df = pd.read_csv(url, sep=' ', names=columns)
    df['target'] = df['target'] - 1  # Convert to 0/1 (originally 1/2)

    # Quick preprocessing: label encode categoricals
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Split and train
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
