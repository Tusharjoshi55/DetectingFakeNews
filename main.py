import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.feature_extraction.text import TfidfVectorizer

news = pd.read_csv('news.csv')  

news.drop("title", axis = 1,inplace = True)

news = news.loc[:, ~news.columns.str.contains('^Unnamed')]

news.isnull().sum()

import re 
def clean_text(text):
    # lower case
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[^\w\s]','',text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

news['text'] = news['text'].apply(clean_text)

x = news['text']
y = news['label']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.4, random_state= 7)
vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

#Train Logistic Regression
modelLG = LogisticRegression()
modelLG.fit(xv_train, y_train)

# Make predictions
y_pred = modelLG.predict(xv_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy :{accuracy:.2f}")
print("Report :\n",classification_report(y_test,y_pred))


def manual_testing(news):
    testing_news = {"text":[news]}
    test_news_df = pd.DataFrame(testing_news)
    test_news_df['text'] = test_news_df['text'].apply(clean_text)
    test_news_x = test_news_df['text']
    test_news_xv = vectorizer.transform(test_news_x)
    pr_LG = modelLG.predict(test_news_xv)

    print("News is :", pr_LG[-1])

check = str(input())
manual_testing(check)