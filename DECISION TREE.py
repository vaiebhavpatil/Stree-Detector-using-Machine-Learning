#!/usr/bin/env python
# coding: utf-8

# # import libaries

# In[1]:


import pandas as pd
import numpy as np


# # reading the datset

# In[2]:


data = pd.read_csv("stress.csv")
data.head()


# # checking null values

# In[3]:


print(data.isnull().sum())


# # removing unwanted characters

# as there are some unwanted characters like special characters / operators in the column
# 
# so we are going to remove those characters

# In[4]:


import nltk
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["text"] = data["text"].apply(clean)


# In[5]:


data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})
data = data[["text", "label"]]
data.head()


# in our label it is mentioned as 0 / 1.
# 
# as our datset is based on stress prediction ( classification problem ), we will be changing the 0/1 to no stress and stress

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

x = np.array(data["text"])
y = np.array(data["label"])

cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, 
                                                test_size=0.33, 
                                                random_state=42)


# we are creating training and testing datset
# 
# as ml model doesn't deal with categorical value
# we are changing to numerical value 
# so it will be easy for our model
# 
# countvectorizer is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text

# # DECISION TREE

# In[7]:


from sklearn import tree


# In[8]:


model = tree.DecisionTreeClassifier()


# In[9]:


model.fit(xtrain,ytrain)


# In[21]:


model.predict(xtest)


# In[13]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)


# In[30]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)


# In[14]:


model.score(xtest,ytest)


# we got accuracy as 63%
