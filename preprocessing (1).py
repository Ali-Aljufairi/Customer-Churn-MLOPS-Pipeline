import os

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

 

# **1. SageMaker Environment Variables**

input_data_path = os.environ['SM_CHANNEL_INPUT']  # Input data location

output_data_path = os.environ['SM_CHANNEL_OUTPUT']  # Output data location

 

# **2. Load the dataset**

dataset = pd.read_csv(os.path.join(input_data_path, 'churn.csv')) 

 

# **3. Preprocessing steps **

# **a. Select relevant features**

features = ['State', 'Account Length', 'Area Code', "Int'l Plan",

    'VMail Plan', 'VMail Message', 'Day Mins', 'Day Calls', 'Day Charge',

    'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls',

    'Night Charge', 'Intl Mins', 'Intl Calls', 'Intl Charge',

    'CustServ Calls', 'Churn?']

dataset = dataset[features]

 

# **b. Encode categorical features**

le = LabelEncoder() 

 

# Encoding columns as needed 

dataset['Int\'l Plan'] = le.fit_transform(dataset['Int\'l Plan'])

dataset['VMail Plan'] = le.fit_transform(dataset['VMail Plan'])

dataset['Churn?'] = le.fit_transform(dataset['Churn?'])

dataset['State'] = le.fit_transform(dataset['State'])

 

# **4. Split data (replace percentages as needed)**

X = dataset.drop('Churn?', axis=1)  # Features

y = dataset['Churn?']  # Target variable

 

# Split into training, validation, and testing sets

X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test   = train_test_split(X_other, y_other, test_size=0.5, random_state=42)

 

# **5. Save datasets**

np.savetxt(os.path.join(output_data_path, "train.csv"), np.column_stack((X_train, y_train)), delimiter=",", fmt='%s')

np.savetxt(os.path.join(output_data_path, "validation.csv"), np.column_stack((X_val, y_val)), delimiter=",", fmt='%s')

np.savetxt(os.path.join(output_data_path, "test.csv"), np.column_stack((X_test, y_test)), delimiter=",", fmt='%s')