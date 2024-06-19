import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv("cleandata.csv")

# Limit data to 1000 rows for training
data_train = data.iloc[:3000]

# Preprocess the dataset (assuming preprocessing steps are already done)
# Assuming you've derived 'hour', 'dayofweek', 'month', and 'year' from the 'DATE OCC' column
data_train['DATE OCC'] = pd.to_datetime(data_train['DATE OCC'])
data_train['hour'] = data_train['DATE OCC'].dt.hour
data_train['dayofweek'] = data_train['DATE OCC'].dt.dayofweek
data_train['month'] = data_train['DATE OCC'].dt.month
data_train['year'] = data_train['DATE OCC'].dt.year

# Define independent and dependent variables
X = data_train[['hour', 'dayofweek', 'month', 'year', 'LAT', 'LON']]
y = data_train['Crime Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear')

# Train the model
svm_classifier.fit(X_train_scaled, y_train)

# Save the trained model using pickle
filename = 'svm_classifier.pkl'
pickle.dump(svm_classifier, open(filename, 'wb'))