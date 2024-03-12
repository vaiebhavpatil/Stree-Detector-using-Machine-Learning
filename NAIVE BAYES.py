#!/usr/bin/env python
# coding: utf-8

# # import libaries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# # reading the datset

# In[2]:


data = pd.read_csv("stress.csv")
data.head()


# # rows and columns

# In[3]:


data.shape


# # checking null values

# In[4]:


print(data.isnull().sum())


# there is no null value in the data set
# 
# so it is good. no need of doing data cleaning

# # removing unwanted characters

# as there are some unwanted characters like special characters / operators in the column
# 
# so we are going to remove those characters

# In[5]:


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


# In[6]:


data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})
data = data[["text", "label"]]
data.head()


# in our label it is mentioned as 0 / 1.
# 
# as our datset is based on stress prediction, we will be changing the 0/1 to no stress and stress

# In[7]:


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

# # NAIVE BAYES

# In[8]:


from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(xtrain, ytrain)


# In[18]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)


# In[10]:


model.score(xtest,ytest)


# we got accuracy as 75%

# In[19]:


model.predict(xtest)


# In[ ]:




