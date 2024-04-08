import streamlit as st
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

stop_words=stopwords.words('english')
puncs=string.punctuation
stemmer=PorterStemmer()

def transform_text(text):
    
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text: 
        if i not in puncs and i not in stop_words:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(stemmer.stem(i))
            
    return ' '.join(y)

CLASS_NAMES=['Not Spam','Spam']

# tfidf=pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open("model.pkl", "rb"))  

st.title("E-Mail/SMS Spam Classifier")

input_sms=st.text_area("Enter the message:")


if st.button("Predict"):

    transformed_sms=transform_text(input_sms)
    transformed_sms=np.expand_dims(transformed_sms,axis=0)

    prediction=model.predict(transformed_sms)[0]
    result=CLASS_NAMES[prediction]

    st.header(result)