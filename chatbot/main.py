# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import random
import time
from gtts import gTTS
from playsound import playsound
import os  
dataset=pd.read_csv("dataset_nlp.tsv",delimiter=".",quoting=3)
corpus_list=[]
nltk.download('stopwords')





def getresponse(dept):
  responses={
    "admin":["Alright raising a ticket for empid:{} working at {} location....an employee from that dept will get in touch with you soon"],
    "about":["You can know more about it by googling than my limited resources"],
    "howto":["you can easily do it using step1 step2 step3"],
    "more":["can you brief me more about it?","i would like to know more about it","please tell me more","subject of beyond my knowledge ..please enlighten me"],
    "sales":["hello {} one of our sales executive will get in touch with you and will contact you at {} within 48 working hours...you can also consider writing us at sales@Xoriant.com"],
    "sysnet_s":["A ticket has been raised for employee id:{} working at {} location....Your issue has been reported to the sysnet software team and they will soon get in touch with you"],
    "sysnet_h":["A ticket has been raised for employee id:{} working at {} location....Your issue has been reported to the sysnet hardware team and they will soon get in touch with you"],
    "ui":["We are extremely sorry for the inconvenience {}....you issue has been reported to the user interface team and they will surely get the issue resolved at {}"],
    "system":["We are extremely sorry for the inconvenience {}....your issue has been reported to the system team and they have already started working on it and will let you at {} once the issue has been resolved "],
    "personal":["this is a personal query and i am sorry, but i cannnot address it"]
  }
  if dept in responses:
    ans=random.choice(responses[dept])
    if dept=='sysnet_h' or dept=='sysnet_s' or dept=='admin':
      empid=int(input("enter your employee id"))
      loc=str(input("enter your work location"))
      ans=ans.format(empid,loc)
    elif dept=='ui' or dept=='system':
      name=input("enter your name please")
      email=input("enter your email id")
      ans=ans.format(name,email)   
    elif dept=='sales':
      name=input("May i please know your name?")
      contact=input("please enter you contact number")
      ans=ans.format(name,contact)
    return ans  

  else:
    return "sorry i dont have any information about it at the moment"  



def train():
  #cleaning the strings
  for i in range(0,len(dataset)):
    compl=re.sub("[^a-zA-Z]"," ",dataset["review"][i])
    compl=compl.lower()
    compl=compl.split()
    ps=PorterStemmer()
    compl =[ps.stem(word) for word in compl if not word in set(stopwords.words("english"))]
    compl=' '.join(compl)
    corpus_list.append(compl)




  #creating bag of words model
  from sklearn.feature_extraction.text import CountVectorizer
  cv=CountVectorizer()
  X=cv.fit_transform(corpus_list).toarray()
  y=dataset.iloc[:,1].values
  y1=y
  #we need to feature code the dependent variable y since the algorithms only work with numerical data
  from sklearn.preprocessing import LabelEncoder, OneHotEncoder
  labelencoder_y = LabelEncoder()
  y = labelencoder_y.fit_transform(y)

  #lets cross validate
  # Splitting the dataset into the Training set and Test set
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


  #now time for the main part lets apply our favourite models
  from sklearn.ensemble import RandomForestClassifier
  forest=RandomForestClassifier(n_estimators=200)
  forest.fit(X_train,y_train)
  #testing the trained models
  ypred=forest.predict(X_test)
  return cv,forest,ps     

def getdept(cv,forest,ps,request):
  compl=re.sub("[^a-zA-Z]"," ",request)
  compl=compl.lower()
  compl=compl.split()
  compl =[ps.stem(word) for word in compl if not word in set(stopwords.words("english"))]
  x=np.zeros(len(cv.get_feature_names()))
  features=list(cv.get_feature_names())
  for i in compl:
      if i in features:
          where=features.index(i)
          x[where]=1
  x=pd.DataFrame(x[:])
  x=x.T
  ans=int(forest.predict(x))
  mylist=list(dataset["dept"].unique())
  mylist.sort()
  return mylist[ans]

def text_to_speech(s):
    tts = gTTS(text=s, lang='en')
    tts.save("good1.mp3")
    playsound('good1.mp3',True)
    os.remove('good1.mp3')
    print(s)
    
      
if __name__=='__main__':
  cv,forest,ps=train()
  tts = gTTS(text="Pepper potts, at your service  How may i help you?", lang='en')
  tts.save("good1.mp3")
  playsound('good1.mp3',True)
  os.remove('good1.mp3')
  print("Pepper potts at your service  How may i help you?")
  while(True):
      
    #taking input from the user  
    human=input()
    
    #checking if saying good bye
    if human.lower() in ["no","nothing","exit","bye","good bye","thank you","thanx","thx"]:
      s="My privelege to serve you .....i wish you a good day!Thank You!"
      text_to_speech(s)
      break
  
    
    
    print("human:{} \n\n".format(human))
    dept=getdept(cv,forest,ps,human)
    time.sleep(1)
    
    #getting the response for the query
    response=getresponse(dept)
    s="{} \n\n".format(response)
    text_to_speech(s)
    time.sleep(1)
    
    #asking if the user has any other queries
    s="Is there anything else i can help you with?"
    text_to_speech(s)
    
  

  






  

  