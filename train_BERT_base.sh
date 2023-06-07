#!/bin/bash
python BERT_base.py 4 64 0 &&
python BERT_base.py 8 64 0 &&
python BERT_base.py 16 64 0 &&

python BERT_base.py 4 128 0 &&
python BERT_base.py 8 128 0 &&
python BERT_base.py 16 128 0 &&

python BERT_base.py 4 64 6 &&
python BERT_base.py 8 64 6 &&
python BERT_base.py 16 64 6 &&

python BERT_base.py 4 128 6 &&
python BERT_base.py 8 128 6 &&
python BERT_base.py 16 128 6 &&

python BERT_base.py 4 64 12 &&
python BERT_base.py 8 64 12 &&
python BERT_base.py 16 64 12 &&

python BERT_base.py 4 128 12 &&
python BERT_base.py 8 128 12 &&
python BERT_base.py 16 128 12