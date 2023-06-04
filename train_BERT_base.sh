#!/bin/bash
python BERT_base.py 8 100 &&
python BERT_base.py 16 100 &&
python BERT_base.py 32 100 &&
python BERT_base.py 64 100

python BERT_base.py 8 256 &&
python BERT_base.py 16 256 &&
python BERT_base.py 32 256 &&
python BERT_base.py 64 100