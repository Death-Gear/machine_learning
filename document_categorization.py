import glob
import errno
import codecs
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

url1 = 'D:/DOC/Document_Categorization/category/accident/*.txt'
url2 = 'D:/DOC/Document_Categorization/category/crime/*.txt'
url3 = 'D:/DOC/Document_Categorization/category/education/*.txt'

target, texts = [], []

file = glob.glob(url1)
for name in file:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            str = " ".join(str.split())
            target.append("accident")
            texts.append(str)


    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

file = glob.glob(url2)
for name in file:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            str = " ".join(str.split())

            target.append("crime")
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

file = glob.glob(url2)
for name in file:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            str = " ".join(str.split())

            target.append("education")
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

df = pd.DataFrame({'texts': texts, 'target': target})
col = ['target', 'texts']
df = df[col]
df = df[pd.notnull(df['texts'])]
df.columns = ['target', 'texts']
df['category_id'] = df['target'].factorize()[0]
category_id_df = df[['target', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'target']].values)
# print(df.tail())

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')
features = tfidf.fit_transform(df.texts).toarray()
labels = df.category_id
# print(features.shape)

N = 2
for target, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(target))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

count_vect = CountVectorizer()
X_data_full_ = count_vect.fit_transform(df['texts'])
tfidf_transformer = TfidfTransformer()
X_data_full_tfidf = tfidf_transformer.fit_transform(X_data_full_)
X_train, X_test, y_train, y_test = train_test_split(X_data_full_tfidf, df['target'], test_size=0.2, random_state=61)

print("naive bayes accuracy is")
clf = MultinomialNB().fit(X_train, y_train)
print(clf.score(X_test, y_test))

print("Decision tree accuracy is")
clf = tree.DecisionTreeClassifier()
clf_output = clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

print("SVM with linear kernel accuracy is")
clf = svm.SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

print("Neural Net Accuracy is")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=61)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

print("KNN accuracy is")
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print(neigh.score(X_test, y_test))
