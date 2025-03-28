# -*- coding: utf-8 -*-
"""Untitled18.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1v9UFckYA5sVDRnFw23_QBmj0K-e9umzv
"""

!pip install kaggle

#configure the path of kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

#api to fetch the dataset from kaggle
!kaggle datasets download -d kazanova/sentiment140

# Extracting the compress dataset
from zipfile import ZipFile
dataset = '/content/sentiment140.zip'

with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('The dataset is extracted')

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

#printing the stop word in english
print(stopwords.words('english'))

#loading the data frame from.csv file to pandas dataframe
twitter_data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv',encoding ='ISO-8859-1')

#checking the number of rows and columns
twitter_data.shape

twitter_data.head()

column_names = ['target','ids','date','flag','user','text']
twitter_data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv',names=column_names, encoding ='ISO-8859-1')

twitter_data.shape

twitter_data.head()
#renaming the attribute in the dataset

#counting the missing values
twitter_data.isnull().sum()

twitter_data['target'].value_counts()
#checking missing values

"""Converting the target "4" ti "1"

"""

twitter_data.replace({'target':{4:1}},inplace=True)

twitter_data['target'].value_counts()
#checking missing values

"""0 ---> negatve tweet,

1 ---> positive tweet
"""

port_stem = nltk.PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)

twitter_data.head()
print(twitter_data['stemmed_content'])
print(twitter_data['target'])

#seperating the data label
X = twitter_data['stemmed_content'].values
y = twitter_data['target'].values

print(X)
print(y)

#splitting the data to training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
print(X_train)
print(X_train)

#converting the textual data to nummerical data

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print(X_train)
print(X_test)

model = LogisticRegression \(max_iter=1000)
model.fit(X_train, y_train)

#model Evaluation

#Accuracy score

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)

print('Accuracy score on the training data :', training_data_accuracy)

X_train_prediction = model.predict(X_test)
training_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on the training data :', test_data_accuracy)

import pickle
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))

#loaded the save model
loaded_model = pickle.load(open('/content/trained_model.sav', 'rb'))

X_new = X_test[200]
print(Y_test[200])

prediction = loaded_model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('Neagtive Tweet')

else:
  print('Positive Tweet')

