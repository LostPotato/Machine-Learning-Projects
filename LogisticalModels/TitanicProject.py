import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the passenger data
passengers = pd.read_csv("passengers.csv")

# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].map({'female': '1', 'male': '0'})

# Fill the nan values in the age column
mean_age = passengers["Age"].mean()  # setting mean_age to replace nan values in age column
passengers["Age"] = passengers["Age"].fillna(mean_age)

# Create a first class feature column
Firstclass = passengers["Pclass"].apply(lambda x: 1 if x == 1 else 0)
passengers['FirstClass'] = Firstclass
# Create a second class feature column
SecondClass = passengers["Pclass"].apply(lambda x: 1 if x == 2 else 0)
passengers['SecondClass'] = SecondClass

# Creating a has cabin column
HasCabin = passengers["Cabin"].apply(lambda x: 0 if x == 0 else 1)
passengers["HasCabin"] = HasCabin

# Select the desired features (Coefficents) and the outcome
features = passengers[["Sex", "Age", "FirstClass", "SecondClass", "HasCabin"]]  # What we think predicts it
survival = passengers[["Survived"]]  # Outcome (What we are trying to predict)

# Perform train, test, split
features_train, features_test, survival_train, survival_test = train_test_split(features, survival)

# Scale the feature data so it has mean = 0 and standard deviation = 1
# Purpose of this scale is to have all coef have the same scale regradless of what they are measuring
scale = StandardScaler()  # Scale object creation
train_features = scale.fit_transform(features_train)
test_features = scale.transform(features_test)

# Create and train the model
model = LogisticRegression()  # Logistic model creation
model.fit(features_train, survival_train)  # fitting model

# Score the model on the train data
train_score = model.score(features_train, survival_train)
print(train_score)
# Score the model on the test data
test_score = model.score(features_test, survival_test)
print(test_score)
# Analyze the coefficients
"""
Sex -- Strong positive 
Age -- tiny negative 
First Class -- Strong Positive 
Second Class -- Relatively Strong positive 
"""
coef_values = model.coef_
print(coef_values)

# Sample passenger features
Jack = np.array([0.0, 20.0, 0.0, 0.0, 0])
Rose = np.array([1.0, 17.0, 1.0, 0.0, 1.0])
You = np.array([0.0, 25.0, 0.0, 1.0, 1.0])

# Combine passenger arrays
Sample = np.array([Jack, Rose, You])

# Scale the sample passenger features
sample = scale.transform(Sample)

# Make survival predictions!
print(model.predict(sample))
print(model.predict_proba(sample))
