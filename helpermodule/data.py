import numpy as np
import pandas as pd
import re

def import_data():
    with open('./data/dark-age-raw-text.txt', 'r') as file:
        raw_data = file.read().replace('\n', ' ')
    return raw_data
    
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

#n_words: # words in an excerpt
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
            

    excerpt_labels = np.array(excerpt_labels)
    excerpt_examples = np.array(excerpt_examples, dtype=object)
    
    return excerpt_labels, excerpt_examples