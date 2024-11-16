import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import pickle  # For saving and loading models
import joblib  # Alternative option for saving models
import os
import logging
import uuid

# Set up logging
logging.basicConfig(filename='model_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Log a simple message
logging.info('Starting Random Forest model training pipeline.')

# Load data
logging.info('Loading data...')
df = pd.read_csv(r"C:\Documents\Thyroid Project\Data\Thyroid_Data.csv")
logging.info('Data loaded successfully.')

# Data Preprocessing Steps
def preprocess_data(df):
    logging.info('Starting data preprocessing...')
    
    # Drop unnecessary columns
    df = df.drop(columns=['Unnamed: 0', 'TBG'])
    
    # Handle missing values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].fillna(df[col].median())
    
    df[['age']] = df[['age']].astype('int64')
    
    # Encode categorical variables
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    combined_encoded = ordinal_encoder.fit_transform(df.select_dtypes(include=['object']).drop(columns=['classes']))
    encoded_features_df = pd.DataFrame(combined_encoded, columns=df.select_dtypes(include=['object']).drop(columns=['classes']).columns)
    df[encoded_features_df.columns] = encoded_features_df
    
    # Encode target
    label_encoder = LabelEncoder()
    df['classes'] = label_encoder.fit_transform(df['classes'])
    
    # Feature Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop(columns=['classes']))
    scaled_df = pd.DataFrame(scaled_features, columns=df.drop(columns=['classes']).columns)
    scaled_df['classes'] = df['classes'].values
    
    logging.info('Data preprocessing completed.')
    return scaled_df, ordinal_encoder, scaler, label_encoder

# Split the dataset into training and test sets
def split_data(scaled_df):
    logging.info('Splitting the data...')
    X = scaled_df.drop(columns=['classes'])
    y = scaled_df['classes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Model Training and Evaluation
def train_and_evaluate(model, X_train_resampled, y_train_resampled, X_test, y_test):
    logging.info(f'Training {model.__class__.__name__} model...')
    
    # Train the model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Log performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f'Model: {model.__class__.__name__}, Accuracy: {accuracy}')
    
    # Print classification report
    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model.__class__.__name__}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Data Preprocessing
scaled_df, encoder, scaler, label_encoder = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(scaled_df)

# Handle class imbalance
logging.info('Handling class imbalance using RandomOverSampler...')
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Initialize Random Forest model
model = RandomForestClassifier(random_state=42)

# Create a unique directory for saving models
model_dir = os.path.join("models", "Random_Forest_Model_" + str(uuid.uuid4()))
os.makedirs(model_dir, exist_ok=True)

# Train and evaluate the model
train_and_evaluate(model, X_train_resampled, y_train_resampled, X_test, y_test)

# Save the Random Forest model locally using `pickle` or `joblib`
model_file_path = os.path.join(model_dir, "random_forest_model.pkl")
with open(model_file_path, 'wb') as f:
    pickle.dump(model, f)
logging.info(f'Model saved locally to {model_file_path}')

# Save the preprocessing objects (encoder, scaler) locally
encoder_file_path = os.path.join(model_dir, "encoder.pkl")
scaler_file_path = os.path.join(model_dir, "scaler.pkl")
label_encoder_file_path = os.path.join(model_dir, "label_encoder.pkl")
with open(encoder_file_path, 'wb') as f:
    pickle.dump(encoder, f)
with open(scaler_file_path, 'wb') as f:
    pickle.dump(scaler, f)
with open(label_encoder_file_path, 'wb') as f:
    pickle.dump(label_encoder, f)
logging.info(f'Preprocessing objects saved locally to {model_dir}')

# Hyperparameter Tuning (Optional)
logging.info('Performing hyperparameter tuning...')
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train_resampled, y_train_resampled)

# Save best Random Forest model
best_rf_model = grid_search.best_estimator_
best_rf_model_file_path = os.path.join(model_dir, "best_random_forest_model.pkl")
with open(best_rf_model_file_path, 'wb') as f:
    pickle.dump(best_rf_model, f)
logging.info(f'Best Random Forest model saved locally to {best_rf_model_file_path}')
