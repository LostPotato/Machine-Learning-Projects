from sklearn.datasets import load_breast_cancer  # importing the dataset
from sklearn.model_selection import train_test_split  # creating training and validation sets
from sklearn.neighbors import KNeighborsClassifier  # making classifier
import matplotlib.pyplot as plt  # graphing data

# loading dataset
breast_cancer_data = load_breast_cancer()

# Examing the dataset
# print(breast_cancer_data.data[0], "\n")
# 1.799e+01 1.038e+01 1.228e+02
# print(breast_cancer_data.target, "\n", breast_cancer_data.target_names)
# 0 0 0 0 0 0 0 ||  'malignant' 'benign'

# Splitting up data
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data,
                                                                                      breast_cancer_data.target,
                                                                                      test_size=.2, random_state=420)

# Setting up the classifier
classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model to training data
classifier.fit(training_data, training_labels)

# View score
print(classifier.score(validation_data, validation_labels))  # K = 3, score = 93%, with random seed = 420

"""
The next bit of code is a look at optimizing neighbors by looping through different sizes of k
"""
# clunky solution for picking the best k value by comparing score values and saving position of value
k_list = range(1, 101)
acc_k = []
acc_score = 0.0
acc_best_holder = 0.0
acc_k_max = 0

# Defining a for loop for k
for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    acc_score = classifier.score(validation_data, validation_labels)
    if acc_score > acc_best_holder:
        acc_best_holder = acc_score
        acc_k_max = k
    acc_k.append(acc_score)

# plotting the accuracy of different K values
plt.plot(k_list, acc_k)
plt.xlabel("k")
plt.ylabel("Validaiton Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()
