from classification import Classification 
from Vectorizer import *
from Model import *
import sys
import csv
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

if __name__ == "__main__":

    classifier = Classification(TF_IDF(),LogisticRegression())
    
    classifier.get_data()
    