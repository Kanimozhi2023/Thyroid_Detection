import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv(r"C:\Documents\Thyroid Project\Data\Thyroid_Data.csv")
print(df)

# Initial check for missing values
print("Initial missing values:\n", df.isnull().sum())

# Data Visualization and Preprocessing
# Compute the correlation matrix
corr_matrix = df.select_dtypes(include='number').corr()
# Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

print(corr_matrix)

# Count the occurrences of each category
bar_data = df['classes'].value_counts()
# Plotting the bar chart
plt.figure(figsize=(6, 4))
bar_data.plot(kind='bar', color='skyblue')
plt.title('Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# Info of the dataset
df.info()
# Shape
print("Shape:", df.shape)
# Describe data
print("Description:", df.describe())

# Drop columns with 100% null values
df = df.drop(columns=['Unnamed: 0', 'TBG'])

# Missing Value handling
print("Total missing value before imputation\n", df.isnull().sum())

# Replace numeric and categorical missing values
# Handling categorical features
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])
# Handling numeric features
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Verify no missing values remain
print("Missing values after imputation:\n", df.isnull().sum())

df[['age']]=df[['age']].astype('int64')
# Outlier Identification
# Check for outliers
fig, axes = plt.subplots(3, 2, figsize=(18, 10))

sns.boxplot(ax=axes[0, 0], data=df, x='classes', y='age')
sns.boxplot(ax=axes[0, 1], data=df, x='classes', y='TSH')
sns.boxplot(ax=axes[1, 0], data=df, x='classes', y='T3')
sns.boxplot(ax=axes[1, 1], data=df, x='classes', y='TT4')
sns.boxplot(ax=axes[2, 0], data=df, x='classes', y='T4U')
sns.boxplot(ax=axes[2, 1], data=df, x='classes', y='FTI')

# Define function to remove outliers
def outliers_removal(numerical_missing):
    numeric_data = numerical_missing.select_dtypes(include='number')
    for column in numeric_data:
        sort = np.sort(numeric_data[column])
        lower_limit, upper_limit = np.percentile(sort, [0, 95])
        detected_outliers = numerical_missing.iloc[np.where((numerical_missing[column] > upper_limit) | (numerical_missing[column] < lower_limit))]
        return detected_outliers

# Call the outliers_removal function with numeric_data
outliers_df = outliers_removal(df)

# Drop outliers
df = df.drop(outliers_df.index)

# Verify no missing values remain after outlier removal
print("Missing values after outlier removal:\n", df.isnull().sum())

# Encoding the categorical data
# Initialize OrdinalEncoder with handle_unknown='use_encoded_value'
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit and transform on combined data
combined_encoded = ordinal_encoder.fit_transform(df.select_dtypes(include=['object']).drop(columns=['classes']))

# Create DataFrame for encoded features
encoded_features_df = pd.DataFrame(combined_encoded, columns=df.select_dtypes(include=['object']).drop(columns=['classes']).columns)

# Replace categorical columns with encoded ones
df[encoded_features_df.columns] = encoded_features_df

# Encode target variable
label_encoder = LabelEncoder()
df['classes'] = label_encoder.fit_transform(df['classes'])

# Verify no missing values are present in the training and test sets before scaling
print("Missing values after outlier removal:\n", df.isnull().sum())

# Handle any remaining missing values explicitly before scaling
numeric_cols = df.select_dtypes(include=['float64']).columns
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Verify no missing values remain after imputation
print("Missing values after outlier removal:\n", df.isnull().sum())


# Feature Scaling
scaler = StandardScaler()

# Fit and transform the data
scaled_features = scaler.fit_transform(df.drop(columns=['classes']))

# Create a DataFrame for the scaled features
scaled_df = pd.DataFrame(scaled_features, columns=df.drop(columns=['classes']).columns)

# Combine the scaled features with the target variable
scaled_df['classes'] = df['classes'].values

# Print the first few rows of the scaled dataframe
print(scaled_df.head())

# Verify the scaling
print("Scaled feature means:\n", scaled_df.drop(columns=['classes']).mean())
print("Scaled feature standard deviations:\n", scaled_df.drop(columns=['classes']).std())

###Feature Selection#####
# Data for Machine learn
# List of columns to drop
# Data for Machine learn
ML_data = scaled_df.drop(columns=['on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
                   'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
                   'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor',
                   'hypopituitary', 'psych'])

ML_data
ML_data.to_csv("preprocessed_data.csv",index=False)

import os

# Check current working directory
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
ML_data = scaled_df 
# Split data into features and target variable
X = ML_data.drop(columns=['classes'])
y = ML_data['classes']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Define a function to train and evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model.__class__.__name__}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(kernel='linear', random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    evaluate_model(model, X_train_resampled, y_train_resampled, X_test, y_test)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Set up the grid search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the model on the resampled data
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Train the best model
best_rf_model = grid_search.best_estimator_

# Evaluate on the test set
y_pred_best_rf = best_rf_model.predict(X_test)
print(f"Best Random Forest Accuracy: {accuracy_score(y_test, y_pred_best_rf)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_best_rf)}")


from sklearn.model_selection import cross_val_score

# Perform 10-fold cross-validation on the best model
cv_scores = cross_val_score(best_rf_model, X_train_resampled, y_train_resampled, cv=10, scoring='accuracy')

print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")

#Saving the model
# After finding the best model
import mlflow.sklearn

# Save the best model
mlflow.sklearn.save_model(best_rf_model, "best_rf_model")

