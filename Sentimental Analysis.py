# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:53:04 2019

@author: user
"""


#import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import nltk
import string
import seaborn as sns
import gc

gc.disable()
folder= 'G:/aclImdb'

labels = {'pos': 1, 'neg': 0}
dataset= pd.DataFrame()

#merging of folders and subfolders
for f in ('test', 'train'):    
    for l in ('pos', 'neg'):
        path = os.path.join(folder, f, l)
        for file in os.listdir (path) :
            with open(os.path.join(path, file),'r', encoding='utf-8') as infile:
                txt = infile.read()
            dataset = dataset.append([[txt, labels[l]]],ignore_index=True)

dataset.columns = ['review','sentiment']

dataset.to_csv('movie_review_analysis', index= False, encoding='utf-8')
dataset.head()
#plotting of frquency of pos and neg sentiments
plot = sns.catplot(x="sentiment", data=dataset, kind="count", height=6, aspect=1.5, palette="bright")
plt.show()

#data cleaning and processing step by step
dataset['reviews'] = dataset['review'].str.replace("[^a-zA-Z]", " ")
dataset['reviews'] = dataset['reviews'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

tokenized_review = dataset['reviews'].apply(lambda x: x.split())
tokenized_review.head()

#join them together
for i in range(len(tokenized_review)):
    tokenized_review[i] = ' '.join(tokenized_review[i])

dataset['reviews'] = tokenized_review

vocabulary = set(tokenized_review)
print(len(vocabulary))

#removing of stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
dataset['reviews']=dataset['reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#stemming done to keep the root of the word
from nltk.stem import PorterStemmer
st = PorterStemmer()
dataset['reviews']=dataset['reviews'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

#formation of wordcloud with all the words in reviews
all_words = ' '.join([text for text in dataset['reviews']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#formation of wordcloud having positive sentiments
positive_words = ' '.join([text for text in dataset['reviews'][dataset['sentiment']==1]])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#formation of worldcloud with negative sentiments
negative_words = ' '.join([text for text in dataset['reviews'][dataset['sentiment']==0]])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#Creating matrix or Bag of words in matrix form
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_features=10000, stop_words='english')
# bag-of-words feature matrix
X = bow_vectorizer.fit_transform(dataset['reviews']).toarray()
y=dataset.iloc[:,1].values

#spliting of dataset into train and test sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.6, random_state=0)

#training model
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(X_train,y_train)

#predicitng values of y for x_test
y_pred=classifier.predict(X_test)

#checking on accuracy and confusion matrix
'''from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_test,y_pred)
confusion_matrix(y_test, y_pred)'''