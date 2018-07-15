import pandas as pd
import numpy as np
import itertools
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('Fake_News_Detection/fake_or_real_news.csv')
df.shape
df.head()

y = df.label
df = df.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.25, random_state=53)

count_vect = CountVectorizer(stop_words='english')
count_train = count_vect.fit_transform(X_train)
count_test = count_vect.transform(X_test)

count_vect.get_feature_names()[:10]
count_df = pd.DataFrame(count_train.A, columns=count_vect.get_feature_names())

tfidf_vect = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vect.fit_transform(X_train)
tfidf_test = tfidf_vect.transform(X_test)

tfidf_vect.get_feature_names()[-10:]
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vect.get_feature_names())

difference = set(count_df.columns) - set(tfidf_df.columns)

count_df.head()
tfidf_df.head()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


clf_nb = MultinomialNB()

clf_nb.fit(count_train, y_train)
predictions = clf_nb.predict(count_test)
score_nb = metrics.accuracy_score(y_test, predictions)
score_nb = score_nb * 100
print("Multinomial Naive Bayes")
print("Accuracy:   %0.3f %%" % score_nb)

from sklearn.svm import SVC

clf_svm = SVC()

clf_svm.fit(count_train, y_train)
predictions = clf_svm.predict(count_test)
score_svm = metrics.accuracy_score(y_test, predictions)
score_svm = score_svm * 100
print("Support Vector Machine")
print("Accuracy is:   %0.3f %%" % score_svm)

from sklearn.tree import DecisionTreeClassifier

clf_dtree = DecisionTreeClassifier(random_state=0)  # this state can be null

clf_dtree.fit(count_train, y_train)
predictions = clf_dtree.predict(count_test)
score_dtree = metrics.accuracy_score(y_test, predictions)
score_dtree = score_dtree * 100
print("Decision Tree")
print("Accuracy is:   %0.3f %%" % score_dtree)

from sklearn.neural_network import MLPClassifier

clf_mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)

clf_mlp.fit(count_train, y_train)
predictions = clf_mlp.predict(count_test)
score_mlp = metrics.accuracy_score(y_test, predictions)
score_mlp = score_mlp * 100
print("Neural Network")
print("Accuracy is:   %0.3f %%" % score_mlp)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
