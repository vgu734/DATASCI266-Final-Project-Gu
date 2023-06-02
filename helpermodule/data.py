import numpy as np
import pandas as pd
import re

#Import data
def import_data():
    with open('./Data/dark-age-raw-text.txt', 'r') as file:
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
            chapter_examples.append(chapter_raw.split('\x0c')[1].replace('\x0c', ''))

    chapter_labels = np.array(chapter_labels)
    print("All Narrators (labels):", np.unique(chapter_labels))

    chapter_examples = np.array(chapter_examples, dtype=object)
    print("\nChapter 1 Narrator:", chapter_labels[0])
    print(chapter_examples[0][:199])
    
    return chapter_labels, chapter_examples