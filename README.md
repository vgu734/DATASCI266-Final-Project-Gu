# DATASCI266-Final-Project-Gu

## Overview
As a model framework, BERT's pre-training on existing 'real' data such as Wikipedia provides an enormous repository
of context and position based word embeddings suitable as a starting point for a myriad of language tasks. Tuning this 
pre-trained model can lead to increased task accuracy for specific tasks and datasets.

However, how does BERT perform when presented with fictional text that differs substantially from the non-fiction
(Wikipedia) data it was trained on? Furthermore, how well can BERT be fine-tuned with fictional text to increase 
performance at a classification text? Finally, how well can BERT pick up subtle variations in word meanings based on 
surrounding fictional context?

To begin answering these questions, I leveraged the base BERT model with the science fiction novel 'Dark Age' by 
Pierce Brown which contains narration from 5 different individuals of vastly differing socio-economic backgrounds. 
The language task involves classifying an excerpt of the text based on which narrator the model believes the text came 
from. I then looked to improve model performance by fine-tuning. Finally, I analyzed various out-of-context words 
specific to the world Pierce Brown created to determine if the fine-tuned BERT could pick up nuanced word meanings,
as represented in word embeddings, based on narrator.

References:
https://drive.google.com/drive/u/1/folders/14x3YKhhHAuhnzXxN8Os1JhcEwuGe9IAt
http://star.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-2989/long_paper43.pdf
https://pubs.aip.org/aip/acp/article-abstract/2605/1/020007/2876057/Exploring-cross-genre-performance-for-age-based?redirectedFrom=fulltext
https://arxiv.org/abs/2010.07245
