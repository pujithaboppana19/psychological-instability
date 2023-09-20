#Importing the required libraries
import numpy as np
import pandas as pd

#Importing the csv dataset
data = pd.read_csv("dataset.csv")
data.head()

#Finding number of rows and columns present in the dataset
data.shape

#Importing NaturalLanguageToolkit, StopWords and PorterStemmer
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Preprocessing the data using Porter Stemmer Algorithm
ps = PorterStemmer()
corpus = []
for i in range(len(X)):
  news = re.sub('[^a-zA-Z]', ' ', X[i])
  news = news.lower()
  news = news.split()

  news = [ps.stem(word) for word in news if word not in stopwords.words('english')]
  news = ' '.join(news)
  corpus.append(news)

#Dividing the dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(np.asarray(corpus), y, test_size = 0.2, random_state = 24)
#Finding the size of training dataset and testing dataset
X_train.shape, X_test.shape

#Importing the Naïve Bayes and Random Forest Classifier from sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

#Training and Testing the model
model = RandomForestClassifier()
x_train = tfidf.fit_transform(X_train)
x_test = tfidf.transform(X_test)

#Printing the accuracy score, precision score, recall score and confusion matrix
pred = pipe_line.predict(X_test)
print(accuracy_score(y_test, pred))
print(precision_score(y_test, pred))
print(recall_score(y_test, pred))
print(confusion_matrix(y_test, pred))

#Pseudocode of app.py
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log
import pandas as pd
import numpy as np



#Function for processing message

def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]   
    return words

#Function for classification
def classify(self, processed_message):
        pDepressive, pPositive = 0, 0
        for word in processed_message:                
            if word in self.prob_depressive:
                pDepressive += log(self.prob_depressive[word])
            else:
                if self.method == 'tf-idf':
                    pDepressive -= log(self.sum_tf_idf_depressive +                  len(list(self.prob_depressive.keys())))
                else:
                    pDepressive -= log(self.depressive_words + len(list(self.prob_depressive.keys())))
            if word in self.prob_positive:
                pPositive += log(self.prob_positive[word])
            else:
                if self.method == 'tf-idf':
                    pPositive -= log(self.sum_tf_idf_positive + len(list(self.prob_positive.keys()))) 
                else:
                    pPositive -= log(self.positive_words + len(list(self.prob_positive.keys())))
            pDepressive += log(self.prob_depressive_tweet)
            pPositive += log(self.prob_positive_tweet)
        return pDepressive >= pPositive

#Function for prediction
def predict(self, testData):
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result

#Predicting sentiment
 analysis = TextBlob(cleanComment(post))
            if analysis.sentiment.polarity > 0:
                sentiment='happy'
            elif analysis.sentiment.polarity == 0:
                sentiment='normal'
            else:
                sentiment='sad'

#Predicting Mental Disorder
 if result:
     mental_flag = True
     message = f"This Post is predicted as Mental Disorder"
else:
       if analysis.sentiment.polarity<0:
            mental_flag = True
            message = f"This Post is predicted as Mental Disorder"
      else:
             non_mental_flag = True
             message = f"This Post is predicted as Normal"
