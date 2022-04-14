# Import libraries.
import os
import pickle

import pandas as pd
import joblib
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


import string
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from random import shuffle
import multiprocessing
from multiprocessing import Pool
import csv

# Create a function to read and prepare data.
def read_data(path, random_state):
   
    true = pd.read_csv(path + "True.csv")
    fake = pd.read_csv(path + "Fake.csv")

    true['label'] = 0
    fake['label'] = 1

    df = pd.concat([true, fake])

    fake = true = None

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df

# Create a function to process text.
def process_text(text):
    
    text = BeautifulSoup(text, "html.parser").get_text()

    text = re.sub('http\S+', 'link', text)

    text = re.sub('\d+', 'number', text)

    text = re.sub('\S+@\S+', 'email', text, flags=re.MULTILINE)
    
    text = text.translate(str.maketrans('', '', string.punctuation))

    text = text.strip()
    
    text = text.lower()
    
    stemmer = SnowballStemmer('english')
    
    words = text.split()
    
    words = [w for w in words if w not in stopwords.words('english')]
    
    words = [stemmer.stem(w) for w in words]
    
    return ' '.join(words)

def prepare_data(data_train, data_test, labels_train, labels_test,
                    cache_dir, cache_file="preprocessed_data.pkl"):
    
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  
    if cache_data is None:
        
        text_train = data_train.progress_apply(process_text)
        text_test = data_test.progress_apply(process_text)

        
        if cache_file is not None:
            cache_data = dict(text_train=text_train, text_test=text_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        text_train, text_test, labels_train, labels_test = (cache_data['text_train'],
                cache_data['text_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return text_train, text_test, labels_train, labels_test

def extract_features(words_train, words_test, vocabulary_size, cache_dir, cache_file="features.pkl"):
    
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = joblib.load(f)
            print("Read features from cache file:", cache_file)
        except:
            pass  
    if cache_data is None:
        
        vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=vocabulary_size,
                                     stop_words='english', analyzer = 'word')
        features_train = vectorizer.fit_transform(words_train).toarray()

        features_test = vectorizer.transform(words_test).toarray()
        
        
        if cache_file is not None:
            vocabulary = vectorizer.vocabulary_
            cache_data = dict(features_train=features_train, features_test=features_test,
                             vocabulary=vocabulary)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                joblib.dump(cache_data, f)
            print("Wrote features to cache file:", cache_file)
    else:
        features_train, features_test, vocabulary = (cache_data['features_train'],
                cache_data['features_test'], cache_data['vocabulary'])
    
    return features_train, features_test, vocabulary
    