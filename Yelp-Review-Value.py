# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:32:50 2020

@author: tyra1
"""
import pandas as pd
import random
import re
from collections import Counter
import numpy as np
import operator
from sklearn import metrics
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')
nltk.download("punkt")

##Loading datasets 
#business_attributes = pd.read_csv(r'C:\Users\tyra1\Desktop\FALL 2020\INSY 446\Project\yelp_business_attributes.csv')
#user = pd.read_csv(r'C:\Users\tyra1\Desktop\FALL 2020\INSY 446\Project\yelp_user.csv')
#business = pd.read_csv(r'C:\Users\tyra1\Desktop\FALL 2020\INSY 446\Project\yelp_business.csv')
#business_hours = pd.read_csv(r'C:\Users\tyra1\Desktop\FALL 2020\INSY 446\Project\yelp_business_hours.csv')
#checkin = pd.read_csv(r'C:\Users\tyra1\Desktop\FALL 2020\INSY 446\Project\yelp_checkin.csv')
review_raw = pd.read_csv(r'C:\Users\tyra1\Desktop\FALL 2020\INSY 446\Project\yelp_review.csv')
#tip = pd.read_csv(r'C:\Users\tyra1\Desktop\FALL 2020\INSY 446\Project\yelp_tip.csv')

review=review_raw

##Adding user affinity (combining useful, funny, cool)
review["user_affinity"]=review.useful+review.funny+review.cool
avg_user_affinity=np.average(review.user_affinity)

##Binary variable creation
one = review.user_affinity > avg_user_affinity
zero = review.user_affinity < avg_user_affinity
review['Binary_comparison'] = np.where(one, 1, np.where(zero, 0, np.NaN))

##Data cleaning
review_isnull=review.isnull()
review.isnull().sum()
review=review.drop(columns=["review_id", "user_id", "business_id", "stars", "date", "useful", "funny", "cool", "user_affinity" ])
review_list = review.values.tolist()
subset_reviews= random.sample(review_list, 50000)

##Function for cleaning text review data
def clean(review):
  review=review.lower()
  review=re.sub(r"[^a-z  $']+", '', review)
  return review

##Separate target and predictor and clean data
X = []
y = []
for review, score in subset_reviews:
  X.append(clean(review))
  y.append(int(score))

##Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=5)

##Create a function that gets the most common words given a size
def vocab(reviews, vocab_size):
  reviews=" ".join(reviews)
  split_words = reviews.split()
  word_count= Counter(split_words).most_common(vocab_size)
  clean_list=[]
  for i in word_count:
    clean_list.append(i[0])
  return clean_list

#Testing best initial num_features where x=1(having a word) will give more prob of the review being valuable

'''max_value=0.065
placeholder=70

##Create common vocab list and remove stop words(common english language words)
for i in range(70,90):
    
    num_features = i
    vocabulary = vocab(X_train, num_features)
    vocabulary = [word for word in vocabulary if not word in stopwords.words()]
    num_features=len(vocabulary)
    num_features
    #print(len(vocabulary))
    #print(vocabulary)
    
    ##Create a function that creates a binary form 1/0 depending if word is in common vocab list or not
    def vectorize(review_string, vocab):
      binary_word=[]
      split_words = review_string.split()
      for e in vocab:
        if e in split_words:
          binary_word.append(1) 
        else:
          binary_word.append(0)
      return binary_word
    
    ##Make binary form of reviews
    X_train_vect=[]  
    for e in X_train:
      X_train_vect.append(vectorize(e, vocabulary))
    
    ##Estimating the Probability Distribution
    Pr_y_1=y_train.count(1)/len(y_train)
    Pr_y_0=y_train.count(0)/len(y_train)
    
    x_is_1_y1=np.zeros(num_features)
    x_is_0_y1=np.zeros(num_features)
    x_is_1_y0=np.zeros(num_features)
    x_is_0_y0=np.zeros(num_features)
    
    
    for obs in range (len(X_train_vect)):
        
        for x_feature in range(num_features):
        
            if ((X_train_vect[obs][x_feature])==1 and (y_train[obs])==1):
                x_is_1_y1[x_feature]=x_is_1_y1[x_feature]+1
                
            elif ((X_train_vect[obs][x_feature])==1 and (y_train[obs])==0):
                x_is_1_y0[x_feature]=x_is_1_y0[x_feature]+1
                
            elif ((X_train_vect[obs][x_feature])==0 and (y_train[obs])==1):
                x_is_0_y1[x_feature]=x_is_0_y1[x_feature]+1
                
            elif ((X_train_vect[obs][x_feature])==0 and (y_train[obs])==0):
                x_is_0_y0[x_feature]=x_is_0_y0[x_feature]+1
            
    x_is_1_y1p=np.divide(x_is_1_y1,y_train.count(1))
    #print(np.average(x_is_1_y1p))
    x_is_0_y1p=np.divide(x_is_0_y1,y_train.count(1))
    np.average(x_is_0_y1p)
    
    x_is_1_y0p=np.divide(x_is_1_y0,y_train.count(0))
    #print(np.average(x_is_1_y0p))
    x_is_0_y0p=np.divide(x_is_0_y0,y_train.count(0))
    np.average(x_is_0_y0p)
    
    test_value=(np.average(x_is_1_y1p)-np.average(x_is_1_y0p))
    if test_value>max_value:
        max_value=test_value
        placeholder=i'''

##best result is 79 initial vocab words
##This is where prob x=1 given y=1 is much greater than prob x=1 given y=0

##Create common vocab list of 16 words and remove stop words(common english language words)
num_features = 79
vocabulary = vocab(X_train, num_features)
vocabulary = [word for word in vocabulary if not word in stopwords.words()]
num_features=len(vocabulary)
num_features
#print(len(vocabulary))
#print(vocabulary)

##Create a function that creates a binary form 1/0 depending if word is in common vocab list or not
def binary_words(review_string, vocab):
  binary_word=[]
  split_words = review_string.split()
  for e in vocab:
    if e in split_words:
      binary_word.append(1) 
    else:
      binary_word.append(0)
  return binary_word

##Make binary form of reviews
X_train_vect=[]  
for e in X_train:
  X_train_vect.append(binary_words(e, vocabulary))

##Estimating the Probability Distribution
Pr_y_1=y_train.count(1)/len(y_train)
Pr_y_0=y_train.count(0)/len(y_train)

x_is_1_y1=np.zeros(num_features)
x_is_0_y1=np.zeros(num_features)
x_is_1_y0=np.zeros(num_features)
x_is_0_y0=np.zeros(num_features)


for obs in range (len(X_train_vect)):
    
    for x_feature in range(num_features):
    
        if ((X_train_vect[obs][x_feature])==1 and (y_train[obs])==1):
            x_is_1_y1[x_feature]=x_is_1_y1[x_feature]+1
            
        elif ((X_train_vect[obs][x_feature])==1 and (y_train[obs])==0):
            x_is_1_y0[x_feature]=x_is_1_y0[x_feature]+1
            
        elif ((X_train_vect[obs][x_feature])==0 and (y_train[obs])==1):
            x_is_0_y1[x_feature]=x_is_0_y1[x_feature]+1
            
        elif ((X_train_vect[obs][x_feature])==0 and (y_train[obs])==0):
            x_is_0_y0[x_feature]=x_is_0_y0[x_feature]+1
        
x_is_1_y1p=np.divide(x_is_1_y1,y_train.count(1))
np.average(x_is_1_y1p)
x_is_0_y1p=np.divide(x_is_0_y1,y_train.count(1))
np.average(x_is_0_y1p)

x_is_1_y0p=np.divide(x_is_1_y0,y_train.count(0))
np.average(x_is_1_y0p)
x_is_0_y0p=np.divide(x_is_0_y0,y_train.count(0))
np.average(x_is_0_y0p)


##Function that creates the Naive Bayes Classifier
def naive_bayes(vec):
    Probability_class1=Pr_y_1
    Probability_class0=Pr_y_0
    #dic=dict{1:Probability_class1, -1: Probability_class0}
    
    for feature in range(len(vec)):
        if (vec[feature])==1:
            Probability_class1=Probability_class1*x_is_1_y1p[feature]
            Probability_class0=Probability_class0*x_is_1_y0p[feature]

        elif (vec[feature])==0:
            Probability_class1=Probability_class1*x_is_0_y1p[feature]
            Probability_class0=Probability_class0*x_is_0_y0p[feature]
            
    dic={1:Probability_class1, 0:Probability_class0}
    
    return max(dic.items(), key=operator.itemgetter(1))[0]

##Measuring accuracy performance
y_train_pred=[] 
for train in X_train_vect:
    #print(naive_bayes(train))
    y_train_pred.append(naive_bayes(train))
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred=[] 
for test in X_test:  
    #print(naive_bayes(test))
    y_test_pred.append(naive_bayes(test))    
print(metrics.accuracy_score(y_test, y_test_pred))

###0.7227714285714286
###0.7355333333333334

##Function to print a review with score
def print_review(review, score):
  print('Review with score of', score,":\n")
  print(review)
  print('\n')

##Testing
test="""Here in Montréal on business (still) and I've been eating well for the most part, but I had a craving for a nice, juicy burger since I've seen this place while walking around downtown, so let's try it! Here's how it went. Atmosphere/Appearance: This place is quaint and tucked right on Rue Crescent. There is only street parking, so good luck with that. Lol. Once inside you're immediately met with the bar on the right and a line of 2-top/4-top tables that can be pushed together on the left. There's a very "homey" vibe here and I felt comfortable. It reminded me of just like, eating in someone's living room. Then you can go upstairs where there's another bar for the busy days and more seating. I really liked it in here. Service: The first time I came here Adrian took care of me and he was awesome. He took great care of me and made me feel so welcome. The 2nd time I came I got takeout, but was treated with the same respect as if I were dining in. Food: My first time I came I got the Brie & Mushroom AAA Bison Burger. LAWD! It was fantastic. It was so juicy, meaty, savory and just amazing. I upgraded my fries to sweet potato fries and they were too... fantastic. Burger Bar KILLED it. So while I was perusing the menu, I noticed they had wings. Y'all know how I feel about wings, so say no more. 2 days after that burger (likely a Wednesday for Wing Wednesday), I came back and got an order or Buffalo Blue wings as well as their Extra Spicy wings. They were so good! The burger was a bit better, but still the wings were fire FOR SURE. I'd totally come back for them. Value: (All prices are in $CAD.) My burger was $24.00 and to add the sweet potato fries was $3.95. My wings were $16.00 each and you get 10 in an order. Very good value for the quality of food that you get. In conclusion, I LOVED this place and it's easily one of my favorite places in Montréal to eat. I will absolutely come back here when I return for future visits."""
#preprocess_sample_point(test, vocabulary)
print_review(test, naive_bayes(binary_words(clean(test),vocabulary)))

test2="""A great Burger place! I ordered the Boss burger and onion rings for take-out. Juicy burger with bacon,tomatoes, American cheese topped with a fried egg. Once you have their burgers you won't go anywhere else. They have everything from burgers to poutine to Mac and Cheese and  milkshakes. Whatever your heart desires they have it. Can't wait to sit inside and try the hang-over burger. It's so big that you have to eat in.. bahaha! Love it!"""
#preprocess_sample_point(test2, vocabulary)
print_review(test, naive_bayes(binary_words(clean(test2),vocabulary)))

#Trying to find which words are most valuable to a high quality review:
res = {vocabulary[i]: x_is_1_y1p[i] for i in range(len(vocabulary))} 
word_prob_dict = dict( sorted(res.items(), key=operator.itemgetter(1),reverse=True))
word_prob_df=pd.DataFrame(word_prob_dict.items(), columns=["Word", "Prob of valuable review"])  
print(word_prob_df)  