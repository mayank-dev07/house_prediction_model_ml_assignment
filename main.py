"""House Price Prediction using Linear Regression
Introduction
The objective of this assignment is to develop a linear regression model to predict house prices using a dataset from Zillow's Zestimate. The dataset includes features such as square footage, number of bedrooms and bathrooms, location, and other attributes of the houses. The model is evaluated using metrics like Mean Squared Error (MSE) and R-squared (R²)."""

"""Step 1: Data Loading and Inspection
First, we load the dataset and perform a preliminary inspection to understand the data structure. We examine the first few rows, data types, and basic statistics of each feature."""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the training and test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Display the first few rows and info about the data
print(train_df.head())
print(train_df.info())
print(train_df.describe())


"""Step 2: Handling Missing Values
To address missing values, we use median imputation for numerical columns and the most frequent value for categorical columns. This helps ensure that the model can be trained effectively without biases introduced by missing data."""




# Check if 'SalePrice' column exists
if 'SalePrice' not in train_df.columns:
    print("Error: 'SalePrice' column not found in the dataset.")
    exit()

# Define the features (X) and target (y)
X = train_df.drop(columns=['SalePrice'])  # Assuming 'SalePrice' is the target column
y = train_df['SalePrice']

# Identify columns with missing values
missing_values = X.isnull().sum().sort_values(ascending=False)
print(missing_values[missing_values > 0])  # Display columns with missing values


"""Step 3: Encoding Categorical Data
Categorical features are identified and converted into numerical values using one-hot encoding. This step is necessary to ensure the regression model can interpret the categorical variables."""



# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', categorical_transformer, X.select_dtypes(include=['object']).columns)
    ])


"""Step 4: Splitting the Data
The data is split into training and validation sets. The training set is used to fit the model, and the validation set is used to evaluate the model’s performance before applying it to the test data."""




# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)


"""Step 5: Training the Linear Regression Model
A linear regression model is trained using the pipeline, which preprocesses the data and fits the model. This step involves combining both numerical and categorical preprocessing."""




# Define the model
model = LinearRegression()

# Create a pipeline to preprocess the data and fit the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Train the model
pipeline.fit(X_train, y_train)


"""Step 6: Model Evaluation
The model is evaluated on the validation set using Mean Squared Error (MSE) and R-squared (R²) metrics. These metrics provide insights into the models accuracy and its ability to explain the variance in house prices."""




# Make predictions on the validation set
y_pred = pipeline.predict(X_valid)

# Evaluate the model
mse = mean_squared_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")


"""Conclusion
In this assignment, we developed a linear regression model to predict house prices based on various features. The model was evaluated using MSE and R², achieving moderate accuracy. Improvements can be explored through feature engineering, advanced models, or hyperparameter tuning."""