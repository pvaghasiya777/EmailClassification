
import  pandas as pd
import numpy as np
import os
import re
import nltk


df = pd.read_csv('emails.csv')
print(df.head())
df.isnull().sum()
df.columns
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_word =set(stopwords.words('english'))
stop_word

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


corpus =[]

for i in range(0,len(df)):
    review = re.sub('[^a-zA-Z0-9]',' ',df['email'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stop_word]
    corpus.append(' '.join(review))
    # print("call from for loop")
    # print(review)
corpus
print(corpus)

df['Category'].value_counts()

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
x_train  = cv.fit_transform(df).toarray()

x_train

x_test  = cv.fit_transform(df).toarray()
x_train.shape
x_test.shape

print(x_train)
print(x_test)

cv.vocabulary

print(cv.vocabulary)
#
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier().fit(x_train,y_train)
#
# y_pred = classifier.predict(x_test)
#
# from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
# accuracy_score(y_test,y_pred)

# confusion_matrix(y_test,y_pred)
#
# print(classification_report(y_test,y_pred))