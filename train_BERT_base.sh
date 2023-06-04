#!/bin/bash
python BERT_base.py 1 &&
python BERT_base.py 4 &&
python BERT_base.py 8 &&
python BERT_base.py 16 &&
python BERT_base.py 32
