import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib

def load_data(filename):
    """Load data from CSV file."""
    return pd.read_csv(filename)

def preprocess_data(data):
    """Preprocess data by separating features and target variables, and encoding labels."""
    X = data.drop(columns=['label'])
    y = data['label']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

    return X, y_onehot, label_encoder

def train_model(X, y):
    """Train a RandomForestClassifier model."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def save_model(model, filename):
    """Save the trained model to disk."""
    joblib.dump(model, filename)

def main():
    # Load the dataset
    data = load_data('../data/crop_data.csv')  # Replace with the actual path

    # Preprocess the data
    X, y_onehot, label_encoder = preprocess_data(data)

    # Train the model
    rf_model = train_model(X, y_onehot)

    # Save the trained model
    save_model(rf_model, '../models/save_model.pkl')  # Replace with desired path and filename

if __name__ == "__main__":
    main()
