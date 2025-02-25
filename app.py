import streamlit as st
import pickle

import nltk

from nltk.tokenize import word_tokenize

# print(word_tokenize("Hello world! This is a test."))

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
# stopwords.words('english')

from nltk.stem.porter import PorterStemmer
import string

ps =PorterStemmer()

def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  ## cloning list
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('tfidf_vectorizer.pkl','rb'))
model = pickle.load(open('mnb_model.pkl','rb'))

st.title("SMS Spam Classifier")

sms = st.text_area("Enter the message")

if st.button('predict'):

    ## preprocess
    transformed_sms = transform(sms)

    ## vectorize
    vector_input = tfidf.transform([transformed_sms])

    ## predict
    final_result = model.predict(vector_input[0])

    ## display
    if final_result==1:
        st.header("spam")
    else:
        st.header("not spam")