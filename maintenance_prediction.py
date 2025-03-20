import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def fetch_data():
    """Fetches LMV maintenance data from SQLite database."""
    conn = sqlite3.connect("releaf_earth.db")
    query = "SELECT * FROM LMV_Maintenance_Tracker;"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def preprocess_data(df):
    """Prepares the dataset for predictive modeling."""
    df = df.dropna()
    df['maintenance_required'] = df['maintenance_type'].apply(lambda x: 1 if x == 'Corrective' else 0)
    X = df[['distance_traveled_km', 'fuel_consumed_liters', 'load_weight_tons', 'previous_maintenance_days']]
    y = df['maintenance_required']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
    """Trains a RandomForest model to predict maintenance needs."""
    X_train, X_test, y_train, y_test = preprocess_data(fetch_data())
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model

def predict_maintenance(model, distance, fuel, load, prev_maint_days):
    """Predicts whether an LMV will need maintenance soon."""
    input_data = np.array([[distance, fuel, load, prev_maint_days]])
    return model.predict(input_data)[0]

if __name__ == "__main__":
    trained_model = train_model()
    sample_prediction = predict_maintenance(trained_model, 500, 50, 10, 30)
    print("Maintenance Required" if sample_prediction == 1 else "Preventive Maintenance Suggested")
