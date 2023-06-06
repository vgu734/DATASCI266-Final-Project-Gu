#!/bin/bash
python BERT_base.py 4 64 &&
python BERT_base.py 8 64 &&
python BERT_base.py 16 64 &&

python BERT_base.py 4 128 &&
python BERT_base.py 8 128 &&
python BERT_base.py 16 128