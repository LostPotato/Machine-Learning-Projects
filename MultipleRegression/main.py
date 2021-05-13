import matplotlib.pyplot as plt
import pandas as pd # Data Handling
from sklearn.model_selection import train_test_split # For model training
from sklearn.linear_model import LinearRegression

"""This is one of my first exercises using mlr. I used the codecadmey exercise as a template to follow"""

# loading in the data set
streeteasy = pd.read_csv("https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/streeteasy.csv")
df = pd.DataFrame(streeteasy)

# Creating a dataframe for indepent variables
x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor',
        'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer',
        'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

# Creating Y Data Frame
y = df[['rent']]

# Creating training set, 80:20 ratio
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = .8, test_size = 0.2, random_state = 6)

# printing out the shape of the data from train_test_split
print(x_train.shape) # (2831, 14)
print(x_test.shape) # (708, 14)

print(y_train.shape)
print(y_test.shape)

# Creating a mlr training model
mlr = LinearRegression()

# Fitting the data to the model
mlr.fit(x_train, y_train)

# Creating prediction dataset
y_predict = mlr.predict(x_test)

# testing the data given a random apartment to predict rent
sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
predict = mlr.predict(sonny_apartment)

print("Predicted rent: $%.2f" % predict) # Actual rent was $2000

# Looking at the predict vs test data
plt.scatter(y_test, y_predict, alpha= .25)
# Adding labels to the data
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Rent vs Predicted Rent")
plt.show()

# examining the weights of non normalized weights of independent factors on data set
print(mlr.coef_)

# Score of the training set
score_train = mlr.score(x_train, y_train)
print(score_train) # r^2 ~ .73
# Score of the test set given the training data
score_test = mlr.score(x_test, y_test) # r^2 ~ .71
print(score_test)

# Playing around with parameters to see if a better R value can be obtained
"""
    For this model I see if certain parameters can be dropped or test size alteration
    The highest with the current iteration is only slightly better having dropped door man
    for the dataset
"""
x_new = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor',
            'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer',
            'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

# Creating Y Data Frame
y_new = df[['rent']]

# Creating training set, 80:20 ratio
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_new, y_new, train_size = .7, test_size = 0.3, random_state = 9)

# Creating a mlr training model
mlr = LinearRegression()

# Fitting the data to the model
mlr.fit(x_train_new, y_train_new)

# Creating prediction dataset
y_predict_new = mlr.predict(x_test_new)

# Looking at the predict vs test data
plt.scatter(y_test_new, y_predict_new, alpha= .25)
# Adding labels to the data
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Rent vs Predicted Rent(Modified Parameters)")
plt.show()

# examining the weights of non normalized weights of independent factors on data set
print(mlr.coef_)

# Score of the training set
score_train = mlr.score(x_train_new, y_train_new)
print(score_train) # r^2 ~ .73
# Score of the test set given the training data
score_test = mlr.score(x_test_new, y_test_new) # r^2 ~ .71
print(score_test)