# LT2222 V23 Assignment 3

NOTE: The work is heavily inspired from the link https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

NOTE: The expectation from a3_features.py is complete, but the code in a3_model.py isn't complete, because I failed to make some smaller changes to adapt the reading from external embedded file. 

1 - The split between 80 20 is accomplished and then the 80 is split to 95, 5. Hence 20 goes to test and 95% of 80 goes to train and 5% of 80 goes to validation data. The stored table contains one dataframe with number of columns equal to 2 + embedding dimension. Of which one column is class describing the author and final column is type telling if the row belongs to train, test or valid. The data is cleaned using regex and rule based to remove potential names.

Model training was successfully accomplished when I first designed the whole script in a single notebook. But after splitting it to features and model, I did not had enough time left to finish the finishing changes to make the model work.

The combined script can be found in this link, https://colab.research.google.com/drive/12_bUl39MdaNIyIDVP3cLdsdhGYX9bdvu?usp=sharing
This file also contains some results at the end and contains no error.

Apologies, for an extremely late submission and fewer documentation. I couldn't as much time as I did in first 2 assignments. 
