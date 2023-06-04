import sys
sys.path.append('helpermodule')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from helpermodule import data
from data import get_chapter_data, get_excerpt_data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Input, Dense, Lambda
from tensorflow.keras.models import Model
from keras.utils import to_categorical
import sklearn as sk

from transformers import BertTokenizer, TFBertModel
from transformers import logging
logging.set_verbosity_error()

#bert_tokenizer = BertTokenizer.from_pretrained('distilbert-base-cased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#bert_model = TFBertModel.from_pretrained('distilbert-base-cased')
bert_model = TFBertModel.from_pretrained('bert-base-cased')

MAX_SEQUENCE_LENGTH = 128                 # set max_length of the input sequence

def format_data(n: int=100):
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

    cls_token = bert_out[0][:, 0, :]

    hidden_1 = tf.keras.layers.Dense(hidden_size, activation='relu', name='hidden_1')(cls_token)
    hidden_1 = tf.keras.layers.Dropout(dropout)(hidden_1)
    hidden_2 = tf.keras.layers.Dense(64, activation='relu', name='hidden_2')(hidden_1)
    hidden_2 = tf.keras.layers.Dropout(dropout)(hidden_2)
    hidden_3 = tf.keras.layers.Dense(32, activation='relu', name='hidden_3')(hidden_2)
    hidden_3 = tf.keras.layers.Dropout(dropout)(hidden_3)


    classification = tf.keras.layers.Dense(5, activation='softmax')(hidden_3)
    
    classification_model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=[classification])
    
    classification_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                 loss='categorical_crossentropy', 
                                 metrics='accuracy')
    
    return classification_model

def main(batch_size: int=1, n: int=100, num_epoch: int=10):
    x_train, x_test, y_train, y_test = format_data(n=n)
    bert_classification_model = create_bert_classification_model(bert_model, num_train_layers=12)

    with tf.device("/GPU:0"):
        bert_classification_model_history = bert_classification_model.fit(
            [x_train.input_ids, x_train.token_type_ids, x_train.attention_mask],
            y_train,
            validation_data=([x_test.input_ids, x_test.token_type_ids, x_test.attention_mask], y_test),
            batch_size=batch_size,
            epochs=num_epoch
        )
    
    
if __name__ == "__main__":
    print(f"Batch size: {sys.argv[1]} Excerpt Length: {sys.argv[2]}")
    main(batch_size=int(list({sys.argv[1]})[0]), n=int(list({sys.argv[2]})[0]))    
