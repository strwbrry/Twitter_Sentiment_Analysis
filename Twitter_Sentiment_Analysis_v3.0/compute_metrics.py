from datasetsHF import load_dataset, ClassLabel, load_metric

import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download(
    ["punkt", "wordnet", "omw-1.4", "averaged_perceptron_tagger", "universal_tagset"]
)

# dataset = load_dataset("vibhorag101/suicide_prediction_dataset_phr")

dtSet = load_dataset('vibhorag101/suicide_prediction_dataset_phr', split='train')
df = dtSet.to_pandas()
df = df.dropna()

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=list(stopset))

y = df.values[:, 1]
X = vectorizer.fit_transform(df.values[:, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1000)

clf = MultinomialNB()
clf.fit(X_train, y_train)


Y_pred = clf.predict(X_test)


acc = accuracy_score(y_test, Y_pred)
f1 = f1_score(y_test, Y_pred, average="binary", pos_label="suicide")
recall = recall_score(y_test, Y_pred, average="binary", pos_label="suicide")
precision = precision_score(y_test, Y_pred, average="binary", pos_label="suicide")


print("Accuracy Score: " + str(acc))
print("F1 Score: " + str(f1))
print("Recall Score: " + str(recall))
print("Precision Score: " + str(precision))