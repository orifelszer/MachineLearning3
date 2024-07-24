import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from car_data_prep import prepare_data
from sklearn.linear_model import ElasticNet
import pickle
import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Load the dataset
data = pd.read_csv('dataset.csv')

# Handle missing values
data = data.dropna(subset=['Price'])  # Drop rows where the target variable is missing
data = prepare_data(data)

# Define feature and target variables
X = data.drop(columns=['Price'])
y = data['Price'].astype(float)

# Identify categorical and numeric features
categorical_features = ['manufactor', 'model', 'Gear', 'Engine_type', 'Color', 'Prev_ownership', 'Curr_ownership']
numeric_features = ['Year', 'Hand', 'capacity_Engine', 'Km', 'Ownership_Change', 'Is_Private', 'Horsepower', 'Is_Automatic']

# Preprocessing pipelines for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Transform the data
X_processed = preprocessor.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train the Elastic Net model
model = ElasticNet(alpha=0.001, l1_ratio=0.1, random_state=42, max_iter=10000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Save the trained pipeline to a pickle file
pickle.dump(model, open('trained_model.pkl', 'wb'))
import joblib

# Assuming you have already fitted your preprocessor on your training data
with open('preprocessor.pkl', 'wb') as file:
    joblib.dump(preprocessor, file)
