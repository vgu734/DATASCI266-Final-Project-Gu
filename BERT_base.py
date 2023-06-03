import sys
sys.path.append('helpermodule')

from helpermodule import data
from data import get_chapter_data, get_excerpt_data
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Input, Dense, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

import sklearn as sk
import os
import nltk
from nltk.data import find
import matplotlib.pyplot as plt
import re

from transformers import BertTokenizer, TFBertModel
from transformers import logging
logging.set_verbosity_error()

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = TFBertModel.from_pretrained('bert-base-cased')

MAX_SEQUENCE_LENGTH = 128                 # set max_length of the input sequence

def format_data():
    print("formatting data")
    excerpt_labels, excerpt_examples = get_excerpt_data(n_words=100)
    x_train, x_test, y_train, y_test = train_test_split(excerpt_examples, excerpt_labels, test_size=.2, random_state=2457)
    
    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)
    
    num_train_examples = 2000      # set number of train examples
    num_test_examples = 500       # set number of test examples

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

def create_bert_classification_model(bert_model,
                                     num_train_layers=0,
                                     hidden_size = 200, 
                                     dropout=0.3,
                                     learning_rate=0.00005):
    """
    Build a simple classification model with BERT. Use the Pooler Output for classification purposes
    """
    if num_train_layers == 0:
        # Freeze all layers of pre-trained BERT model
        bert_model.trainable = False

    elif num_train_layers == 12: 
        # Train all layers of the BERT model
        bert_model.trainable = True

    else:
        # Restrict training to the num_train_layers outer transformer layers
        retrain_layers = []

        for retrain_layer_number in range(num_train_layers):

            layer_code = '_' + str(11 - retrain_layer_number)
            retrain_layers.append(layer_code)
          
        
        print('retrain layers: ', retrain_layers)

        for w in bert_model.weights:
            if not any([x in w.name for x in retrain_layers]):
                #print('freezing: ', w)
                w._trainable = False

    input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int64, name='input_ids_layer')
    token_type_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int64, name='token_type_ids_layer')
    attention_mask = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int64, name='attention_mask_layer')

    bert_inputs = {'input_ids': input_ids,
                   'token_type_ids': token_type_ids,
                   'attention_mask': attention_mask}      

    bert_out = bert_model(bert_inputs)

    pooler_token = bert_out[1]
    #cls_token = bert_out[0][:, 0, :]

    hidden = tf.keras.layers.Dense(hidden_size, activation='relu', name='hidden_layer')(pooler_token)


    hidden = tf.keras.layers.Dropout(dropout)(hidden)  


    classification = tf.keras.layers.Dense(1, activation='sigmoid',name='classification_layer')(hidden)
    
    classification_model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=[classification])
    
    classification_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), 
                                 metrics='accuracy')
    
    return classification_model

def main(batch_size: int=1, num_epoch: int=2):
    x_train, x_test, y_train, y_test = format_data()
    bert_classification_model = create_bert_classification_model(bert_model, num_train_layers=12)
    #confirm all layers are frozen
    bert_classification_model.summary()
    
    bert_classification_model_history = bert_classification_model.fit(
        [x_train.input_ids, x_train.token_type_ids, x_train.attention_mask],
        y_train,
        validation_data=([x_test.input_ids, x_test.token_type_ids, x_test.attention_mask], y_test),
        batch_size=batch_size,
        epochs=2
    ) 
    
    
if __name__ == "__main__":
    print("in main")
    main()
    print("all done")
    