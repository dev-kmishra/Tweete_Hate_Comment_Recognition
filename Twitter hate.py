#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np
import os, re


# #### Read in the csv using pandas 

# In[2]:


inp_tweets0 = pd.read_csv("TwitterHate.csv")
inp_tweets0.head()


# In[3]:


inp_tweets0.label.value_counts(normalize=True)


# In[4]:


inp_tweets0.tweet.sample().values[0]


# #### Get the tweets into a list, for easy text clean up and manipulation

# In[5]:


tweets0 = inp_tweets0.tweet.values


# In[6]:


len(tweets0)


# In[7]:


tweets0[:5]


# The tweets contain - 
# 1. URLs
# 2. Hashtags
# 3. User handles
# 4. 'RT'

# ## Cleanup 

# #### Normalizing case

# In[8]:


tweets_lower = [twt.lower() for twt in tweets0]


# In[9]:


tweets_lower[:5]


# #### Remove user handles, begin with '@'

# In[10]:


import re


# In[11]:


re.sub("@\w+","", "@Rahim this course rocks! http://rahimbaig.com/ai")


# In[12]:


tweets_nouser = [re.sub("@\w+","", twt) for twt in tweets_lower]


# In[13]:


tweets_nouser[:5]


# #### Remove URLs

# In[14]:


re.sub("\w+://\S+","", "@Rahim this course rocks! http://rahimbaig.com/ai")


# In[15]:


tweets_nourl = [re.sub("\w+://\S+","", twt) for twt in tweets_nouser]


# In[16]:


tweets_nourl[:5]


# #### Tokenze using Tweet Tokenizer from NLTK

# In[17]:


from nltk.tokenize import TweetTokenizer


# In[18]:


get_ipython().run_line_magic('pinfo', 'TweetTokenizer')


# In[19]:


tkn = TweetTokenizer()


# In[20]:


print(tkn.tokenize(tweets_nourl[0]))


# In[21]:


tweet_token = [tkn.tokenize(sent) for sent in tweets_nourl]
print(tweet_token[0])


# ### Remove punctuations and stop words and other redundant terms tike 'rt', 'amp'
# - Also remove hashtags

# In[22]:


from nltk.corpus import stopwords
from string import punctuation


# In[23]:


stop_nltk = stopwords.words("english")
stop_punct = list(punctuation)


# In[24]:


stop_punct.extend(['...','``',"''",".."])


# In[25]:


stop_context = ['rt', 'amp']


# In[26]:


stop_final = stop_nltk + stop_punct + stop_context


# #### Function to 
# - remove stop words from a single tokenized sentence
# - remove # tags
# - remove terms with length = 1

# In[27]:


def del_stop(sent):
    return [re.sub("#","",term) for term in sent if ((term not in stop_final) & (len(term)>1))]


# In[28]:


del_stop(tweet_token[4])


# In[29]:


tweets_clean = [del_stop(tweet) for tweet in tweet_token]


# #### Check out the top terms in the tweets

# In[30]:


from collections import Counter


# In[31]:


term_list = []
for tweet in tweets_clean:
    term_list.extend(tweet)


# In[32]:


res = Counter(term_list)
res.most_common(10)


# ## Data formatting for predictive modeling 

# #### Join the tokens back into strings

# In[33]:


tweets_clean[0]


# In[34]:


tweets_clean = [" ".join(tweet) for tweet in tweets_clean]


# In[35]:


tweets_clean[0]


# ### Separate X and Y and perform train test split, 70-30

# In[36]:


len(tweets_clean)


# In[37]:


len(inp_tweets0.label)


# In[38]:


X = tweets_clean
y = inp_tweets0.label.values


# ####  Train test split

# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)


# ### Create a document term matrix using count vectorizer

# In[40]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[41]:


vectorizer = TfidfVectorizer(max_features = 5000)


# In[42]:


len(X_train), len(X_test)


# In[43]:


X_train_bow = vectorizer.fit_transform(X_train)

X_test_bow = vectorizer.transform(X_test)


# In[44]:


X_train_bow.shape, X_test_bow.shape


# ### Model building

# ### Using a *simple* Logistic Regression

# In[48]:


from sklearn.linear_model import LogisticRegression


# In[49]:


logreg = LogisticRegression()


# In[50]:


logreg.fit(X_train_bow, y_train)


# In[51]:


y_train_pred = logreg.predict(X_train_bow)
y_test_pred = logreg.predict(X_test_bow)


# In[52]:


from sklearn.metrics import accuracy_score, classification_report


# In[53]:


accuracy_score(y_train, y_train_pred)


# In[54]:


print(classification_report(y_train, y_train_pred))


# #### Adjusting for class imbalance

# In[55]:


logreg = LogisticRegression(class_weight="balanced")


# In[56]:


logreg.fit(X_train_bow, y_train)


# In[57]:


y_train_pred = logreg.predict(X_train_bow)
y_test_pred = logreg.predict(X_test_bow)


# In[58]:


accuracy_score(y_train, y_train_pred)


# In[59]:


print(classification_report(y_train, y_train_pred))


# In[60]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold


# In[63]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'C': [0.01,0.1,1,10,100],
    'penalty': ["l1","l2"]
}


# In[64]:


get_ipython().run_line_magic('pinfo', 'LogisticRegression')


# In[65]:


classifier_lr = LogisticRegression(class_weight="balanced")


# In[66]:


# Instantiate the grid search model
grid_search = GridSearchCV(estimator = classifier_lr, param_grid = param_grid, 
                          cv = StratifiedKFold(4), n_jobs = -1, verbose = 1, scoring = "recall" )


# In[69]:


grid_search.fit(X_train_bow, y_train)


# In[71]:


grid_search.best_estimator_


# ### Using the best estimator to make predictions on the test set

# In[72]:


y_test_pred = grid_search.best_estimator_.predict(X_test_bow)


# In[73]:


y_train_pred = grid_search.best_estimator_.predict(X_train_bow)


# In[74]:


print(classification_report(y_test, y_test_pred))


# In[ ]:




