
# coding: utf-8

# In[214]:


get_ipython().magic(u'matplotlib inline')

import numpy as np
from sklearn.model_selection import train_test_split
from pylab import *
import pandas as pd
from sklearn import linear_model

df = pd.read_csv("inter2.csv")


# In[275]:


## For extracting the features from the csv file

import csv
import sys

f = open("inter2.csv", "rb")
reader = csv.reader(f)
target=[]
k=0
for row in reader:
    if(k>0):
        temp=[]
        for i in row[1:7]:
            temp.append(int(i))
        target.append(temp)
    k+=1    

print target


# In[199]:


## Forr opening that file

import pickle
f=open('text_answer.txt','r')
new_text=pickle.load(f)


# In[200]:


## For storing variables in a data file as text format

import pickle
f=open('text_answer.txt','w')
pickle.dump(text,f)
f.close()


# In[201]:


new_text[1]=new_text[1].replace("  "," ")


# In[202]:


"Hello. And".split(" ")


# In[204]:


import nltk
res=nltk.pos_tag(new_text[1].split(" "))
print res


# In[205]:


tags_list=[0]*len(res)
for i in range(len(res)):
    tags_list[i]=res[i][1]
print tags_list


# In[222]:


tags_list=list(sorted(set(tags_list)))
print tags_list


# In[223]:


t=['EX','FW','JJR','JJS','LS','MD','NNPS','PDT','POS','RBR','RBS','SYM','UH','VBP','WP$','WDT',',']


# In[228]:


tags_final=sorted([',','CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']) 
print tags_final


# In[232]:


import collections

inputs = []

for text in new_text:
    
    #print text
    text = text.strip()
    text=text.replace("  "," ")
    
    #print(len(nltk.pos_tag(text.split(" "))))
    res=nltk.pos_tag(text.split(" "))
    
#     print res
    p=len(res)
    #print p
    tags_list=[0]*p
    for j in range(p):
        tags_list[j]=res[j][1]

    tags =collections.OrderedDict()
    for k in range(len(tags_final)):
        tags[tags_final[k]] = 0


    for l in range(len(tags_list)):
        tags[tags_list[l]]+=1
        
    inputs.append(tags.values())
    #print(tags.values())
    
print len(inputs)



# In[274]:


len(target)


# In[238]:


import pickle
f=open('input_features.txt','w')
pickle.dump(inputs,f)
f.close()


# In[237]:


len(inputs[27])


# In[ ]:


#nltk.download('tagsets')


# In[ ]:


#nltk.download('averaged_perceptron_tagger')


# In[ ]:


df.head()


# In[ ]:


text = df["Transcription"]
print text


# In[ ]:


#quantify answer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack

# For beginning, transform train['FullDescription'] to lowercase using text.lower()
df['Transcription'].str.lower()

# Then replace everything except the letters and numbers in the spaces.
# it will facilitate the further division of the text into words.
df['Transcription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

# Convert a collection of raw documents to a matrix of TF-IDF features with TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5)
X_tfidf = vectorizer.fit_transform(df['Transcription'])

print X_tfidf


# In[292]:


#split to test data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier


X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.25, random_state=42)


# In[293]:


len(y_train[0])


# In[297]:


# fit a model
lm =  linear_model.LinearRegression()
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
print(predictions)

# error
error=np.mean(abs(predictions-y_test))
print"Error is ",error


# In[ ]:


lm.predict(feature_values)


# In[304]:


for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        predictions[i][j]=round(predictions[i][j])


# In[300]:


import math
math.ceil(predictions[0])


# In[305]:


# error
error=np.mean(abs(predictions-y_test))
print"Error is ",error


# In[306]:


predictions


# In[308]:


y_test


# In[311]:


t=abs(predictions-y_test)
print t


# In[312]:


sum(t)


# In[316]:


f=37.0/(6*7)


# In[317]:


f

