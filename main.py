

option1 = input('Which model do you wish to run?\na) Lexical Based\nb) Word Embedding based\nc) Bag of Words based\n')

from preprocess import *


if option1.lower() == 'a':
    print("Running model based on Langrouh and McDonald lexical database...\n\n")
    from lexical import *

elif option1.lower() == 'b':
    pass

elif option1.lower() == 'c':
    
    pass
