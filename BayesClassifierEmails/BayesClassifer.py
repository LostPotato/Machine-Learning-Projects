from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

"""

The first section of this program demonstrates an attempt to find a really accurate model. The second part is 
attempt to make a terrible classifier. The third part is looking at the overall ability given the whole dataset (Cause 
the program to take a while depending on gpu) I also maintain the models throughout the program and its fun putting 
random data into the smaller models (try sci topics in talk show one) 

"""

# loading in the dataset from sklearn and selecting catagories to compare
emails = fetch_20newsgroups()

# examing data set of emails
# print(emails.data[5]) # hockey
# print(emails.target[5]) # one -- hockey

# making the training and test set of data
train_emails = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'], subset='train', shuffle=True,
                                  random_state=108)
test_emails = fetch_20newsgroups(categories=['rec.sport.baseball', 'rec.sport.hockey'], subset='test', shuffle=True,
                                 random_state=108)

# Transforming the emails in to count vectors
counter = CountVectorizer()  # making the object
counter.fit(test_emails.data + train_emails.data)  # finding the possible words that can be found
train_counts = counter.transform(train_emails.data)  # Get counts of words in train set
test_counts = counter.transform(test_emails.data)  # Get counts of words in test set

# Making the classifier (Naive Bayes Classifier)
classifier = MultinomialNB()  # Bayes object
classifier.fit(train_counts, train_emails.target)  # Training it on train set

# examining the score of the model
print(classifier.score(test_counts, test_emails.target))  # With randomstate = 108, model accuracy 97%

"""
Part 2 -- For getting the model confused, it might seem counter to what you think (at least for me), but
put the most similar ones together because more langauge will be shared across them
"""

# making the training and test set of data
train_emails_1 = fetch_20newsgroups(categories=['comp.os.ms-windows.misc', 'comp.windows.x'], subset='train',
                                    shuffle=True, random_state=108)
test_emails_1 = fetch_20newsgroups(categories=['comp.os.ms-windows.misc', 'comp.windows.x'], subset='test',
                                   shuffle=True, random_state=108)

# Transforming the emails in to count vectors
counter_1 = CountVectorizer()  # making the object
counter_1.fit(test_emails_1.data + train_emails_1.data)  # finding the possible words that can be found
train_counts_1 = counter_1.transform(train_emails_1.data)  # Get counts of words in train set
test_counts_1 = counter_1.transform(test_emails_1.data)  # Get counts of words in test set

# Making the classifier (Naive Bayes Classifier)
classifier_1 = MultinomialNB()  # Bayes object
classifier_1.fit(train_counts_1, train_emails_1.target)  # Training it on train set

# examining the score of the model
print(classifier_1.score(test_counts_1, test_emails_1.target))  # With randomstate = 108, model accuracy 50%

"""
Part 3 -- full model
"""
# making the training and test set of data
train_emails_2 = fetch_20newsgroups(subset='train', shuffle=True, random_state=108)
test_emails_2 = fetch_20newsgroups(subset='test', shuffle=True, random_state=108)

# Transforming the emails in to count vectors
counter_2 = CountVectorizer()  # making the object
counter_2.fit(test_emails_2.data + train_emails_2.data)  # finding the possible words that can be found
train_counts_2 = counter_2.transform(train_emails_2.data)  # Get counts of words in train set
test_counts_2 = counter_2.transform(test_emails_2.data)  # Get counts of words in test set

# Making the classifier (Naive Bayes Classifier)
classifier_2 = MultinomialNB()  # Bayes object
classifier_2.fit(train_counts_2, train_emails_2.target)  # Training it on train set

# examining the score of the model
print(classifier_2.score(test_counts_2, test_emails_2.target))  # With randomstate = 108, model accuracy 76%
