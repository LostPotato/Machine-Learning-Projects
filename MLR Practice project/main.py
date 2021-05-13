import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# loading the data set in
df = pd.read_csv("tennis_stats.csv")

# Examing some of the relationships between data
plt.scatter(df["BreakPointsFaced"], df["Winnings"]) # Strong positive relaitonship
plt.show()
plt.clf()
# Do Breakpoints conversion influence winning amount?
plt.scatter(df["BreakPointsOpportunities"], df["Winnings"]) # No not really
plt.show()
plt.clf()
# does First service  influence service games won?
plt.scatter(df["Aces"], df["Winnings"]) # Slight Positive Relationship
plt.show()
plt.clf()

# Linear model
reg = LinearRegression()
Mlr = LinearRegression()

# organizing daat for model training I
x_I = df[["BreakPointsFaced"]]
y_I = df[["Winnings"]]

# model II
x_II = df[["BreakPointsOpportunities"]]
y_II = df[["Winnings"]]

# model III
x_III = df[["BreakPointsOpportunities","Aces"]]
y_III = df[["Winnings"]]

# Splitting data into training set
x_I_training, x_I_test, y_I_training, y_I_test = train_test_split(x_I, y_I, train_size = .8, random_state = 82)
x_II_training, x_II_test, y_II_training, y_II_test = train_test_split(x_II, y_II, train_size = .8, random_state = 82)
x_III_training, x_III_test, y_III_training, y_III_test = train_test_split(x_III, y_III, train_size = .8, random_state = 82)

# Fitting the model
model_I = reg.fit(x_I_training, y_I_training)
model_II = reg.fit(x_II_training, y_II_training)
model_III = Mlr.fit(x_III_training, y_III_training)

# Evulating the score of the model on train and test
train_i_score = model_I.score(x_I_training, y_I_training)
test_i_score = model_I.score(x_I_test, y_I_test)
# Model II
train_ii_score = model_II.score(x_II_training, y_II_training)
test_ii_score = model_II.score(x_II_test, y_II_test)
# Model III
train_iii_score = model_III.score(x_III_training, y_III_training)
test_iii_score = model_III.score(x_III_test, y_III_test)
# Veiwing score
print(train_i_score, test_i_score)
print(train_ii_score, test_ii_score)
print(train_iii_score, test_iii_score)
# Viewing line
prediction_i = model_II.predict(x_I_test)
prediction_ii = model_II.predict(x_II_test)
prediction_iii = model_III.predict(x_III_test)
# Model View
plt.scatter(y_I_test, prediction_i, alpha = .7)
plt.show()
plt.clf()
plt.scatter(y_II_test, prediction_ii, alpha = .7)
plt.show()
plt.clf()
plt.scatter(y_III_test, prediction_iii, alpha = .7)
plt.show()
plt.clf()
