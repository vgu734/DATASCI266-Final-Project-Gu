import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from transformers import BertTokenizer, TFBertModel
from transformers import logging

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

"""
Method to read in the raw text data
"""
def import_data():
    with open('./data/dark-age-raw-text.txt', 'r') as file:
        raw_data = file.read().replace('\n', ' ')
    return raw_data

"""
Method that returns chapter level labeled data (not used in code)
"""
def get_chapter_data():
    chapter_labels = []
    chapter_examples = []

    raw_data = import_data()
    chapters_raw = np.array(raw_data.split('***'))
    for chapter_raw in chapters_raw:
        if len(chapter_raw) > 0:
            chapter_labels.append(chapter_raw.split('\x0c')[0])
            chapter_examples.append(' '.join(chapter_raw.split('\x0c')[1:]).replace('\x0c', ' '))

    chapter_labels = np.array(chapter_labels)
    chapter_examples = np.array(chapter_examples, dtype=object)
    
    return chapter_labels, chapter_examples

"""
Method that returns excerpt level labeled data in raw human readable form
params: n_words: # words in an excerpt
"""
def get_excerpt_data(n_words: int = 100):
    excerpt_labels = []
    excerpt_examples = []

    raw_data = import_data()
    chapters_raw = np.array(raw_data.split('***'))
    for chapter_raw in chapters_raw:
        if len(chapter_raw) > 0:
            words = ' '.join(chapter_raw.split('\x0c')[1:]).replace('\x0c', ' ').split()
            for i in range(0, len(chapter_raw), n_words):
                if len(' '.join(words[i:i+n_words])) > 1:
                    excerpt_labels.append(chapter_raw.split('\x0c')[0])
                    excerpt_examples.append(' '.join(words[i:i+n_words]))
                else:
                    break
            
    #label_dct = {'Darrow':1, 'Ephraim':2, 'Lyria':3, 'Lysander':4, 'Virginia':5}
    #excerpt_labels = [label_dct[key] for key in excerpt_labels]
    
    return excerpt_labels, excerpt_examples

"""
Method that returns excerpt level labeled data in BERT tokenized form ready for model ingest
params: n: # words in an excerpt
"""
def format_data(n: int=100):
    #bert_tokenizer = BertTokenizer.from_pretrained('distilbert-base-cased')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    #bert_model = TFBertModel.from_pretrained('distilbert-base-cased')
    bert_model = TFBertModel.from_pretrained('bert-base-cased')
    
    MAX_SEQUENCE_LENGTH = 128 
    
    excerpt_labels, excerpt_examples = get_excerpt_data(n_words=n)
    x_train, x_test, y_train, y_test = train_test_split(excerpt_examples, excerpt_labels, test_size=.2, random_state=2457)
    
    num_train_examples = len(x_train)      # set number of train examples
    num_test_examples = len(y_train)       # set number of test examples
    
    label = preprocessing.LabelEncoder()
    y_train = label.fit_transform(y_train)
    y_train = to_categorical(y_train)
    y_test = label.fit_transform(y_test)
    y_test = to_categorical(y_test)

    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)

    all_train_examples = [x.decode('utf-8') for x in x_train.numpy()]
    all_test_examples = [x.decode('utf-8') for x in x_test.numpy()]
    
    x_train = bert_tokenizer(all_train_examples[:num_train_examples],
                  max_length=MAX_SEQUENCE_LENGTH,
                  truncation=True,
                  padding='max_length', 
                  return_tensors='tf')
    y_train = y_train[:num_train_examples]

    x_test = bert_tokenizer(all_test_examples[:num_test_examples],
                  max_length=MAX_SEQUENCE_LENGTH,
                  truncation=True,
                  padding='max_length', 
                  return_tensors='tf')
    y_test = y_test[:num_test_examples]
    
    return x_train, x_test, y_train, y_test