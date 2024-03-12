#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("stress.csv")
data.head()


# In[3]:


print(data.isnull().sum())


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


# In[7]:


from sklearn.linear_model import LogisticRegression


# In[8]:


model = LogisticRegression()


# In[26]:


model.fit(xtrain,ytrain)


# In[28]:


model.predict(xtest)


# In[40]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)


# In[24]:


model.score(xtest,ytest)


# we got accuracy as 71%

# In[ ]:




