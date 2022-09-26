# Python libraries
import datetime as dt
import re
import pickle
from tqdm.notebook import tqdm
import os
import sys
import time
import random
import json
from collections import defaultdict, Counter

# Data Science modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
plt.style.use('ggplot')

# Import Scikit-learn moduels

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import svm 
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, plot_confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest,chi2 

# import scikitplot as skplt

# Import nltk modules and download dataset
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from preprocess import stopwords, create_corpus, get_frequent_words, df, corpus

from build_model import Build_Model

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Set Random Seed
random.seed(42)
np.random.seed(42)
rand_seed = 42
random_state =42

# Set Seaborn Style
sns.set(style='white', context='notebook', palette='deep')

result_df = pd.DataFrame(columns=['Accuracy', 'Macro Averaged F1'], index=['A: Lexicon', 'B: Tf-idf'])

# Define metrics
# Here, use F1 Macro to evaluate the model.
def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1

scoring = {'Accuracy': 'accuracy', 'F1': 'f1_macro'}
refit = 'F1'

kfold = StratifiedKFold(n_splits=10)

train_df = df
num_words_per_sentence = train_df['sentences'].apply(lambda x: len(nltk.word_tokenize(x)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))

sns.countplot(x='labels', data=train_df, ax=ax1)
ax1.set_title('Sentiment Distribution', fontsize=16)
ax2.hist(num_words_per_sentence,bins = 16)
ax2.set_xlabel('The number of words per sentence')
ax2.set_title('Words Distribution', fontsize=16)
# plt.show(block = True);

# Encode the label
le = LabelEncoder()
le.fit(train_df['labels'])
train_df['labels'] = le.transform(train_df['labels'])
# list(le.inverse_transform(train_df['label']))
train_df['labels']

# Check most frequent words which are not in stopwords
counter = Counter(corpus)
most = counter.most_common()[:60]
x, y = [], []
for word, count in most:
    if word not in stopwords:
        x.append(word)
        y.append(count)

plt.title("Most frequent words")
plt.figure(figsize=(15,7))
sns.barplot(x=y, y=x);
# plt.show();

# Load sentiment data
sentiment_df = pd.read_csv('LM-SA-2020.csv')

# Make all words lower case
sentiment_df['word'] = sentiment_df['word'].str.lower()
sentiments = sentiment_df['sentiment'].unique()
sentiment_df.groupby(by=['sentiment']).count()

# key (sentiment) : value (words)
sentiment_dict = { sentiment: sentiment_df.loc[sentiment_df['sentiment']==sentiment]['word'].values.tolist() for sentiment in sentiments}

# Consider Negation
negate = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt", "ain't", "aren't", "can't",
          "couldn't", "daren't", "didn't", "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt",
          "neither", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "neednt", "needn't",
          "never", "none", "nope", "nor", "not", "nothing", "nowhere", "oughtnt", "shant", "shouldnt", "wasnt",
          "werent", "oughtn't", "shan't", "shouldn't", "wasn't", "weren't", "without", "wont", "wouldnt", "won't",
          "wouldn't", "rarely", "seldom", "despite", "no", "nobody"]

def negated(word):
    """
    Determine if preceding word is a negation word
    """
    if word.lower() in negate:
        return True
    else:
        return False

def tone_count_with_negation_check(dict, article):
    """
    Count positive and negative words with negation check. Account for simple negation only for positive words.
    Simple negation is taken to be observations of one of negate words occurring within three words
    preceding a positive words.
    """

    polarity = 0
    pos_count = 0
    neg_count = 0
    tone_score = 0

    pos_words = []
    neg_words = []
 
    input_words = re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', article.lower()) #extracting all words
    word_count = len(input_words)
     
    for i in range(0, word_count):
        if input_words[i] in dict['Negative']:

            if i >= 3:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]) or negated(input_words[i - 3]):
                    pos_count += 1
                    pos_words.append(input_words[i] + ' (with negation)')
                else:
                    neg_count += 1
                    neg_words.append(input_words[i])
            elif i == 2:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]):
                    pos_count += 1
                    pos_words.append(input_words[i] + ' (with negation)')
                else:
                    neg_count += 1
                    neg_words.append(input_words[i])
            elif i == 1:
                if negated(input_words[i - 1]):
                    pos_count += 1
                    pos_words.append(input_words[i] + ' (with negation)')
                else:
                    neg_count += 1
                    neg_words.append(input_words[i])
            elif i == 0:
                neg_count += 1
                neg_words.append(input_words[i])
        if input_words[i] in dict['Positive']:
            if i >= 3:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]) or negated(input_words[i - 3]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 2:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 1:
                if negated(input_words[i - 1]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 0:
                pos_count += 1
                pos_words.append(input_words[i])
 
    if word_count > 0:
        tone_score = 100 * (pos_count - neg_count) / word_count

        if not(pos_count == 0 and neg_count == 0): 
            polarity = (pos_count - neg_count)/(pos_count + neg_count)
        else:
            polarity = 0
    else:
        polarity = 0
        tone_score = 0
    
    
    results = [polarity, tone_score, word_count, pos_count, neg_count, pos_words, neg_words]
 
    return results

columns = ['polarity', 'tone_score', 'word_count', 'n_pos_words', 'n_neg_words', 'pos_words', 'neg_words']

# Analyze tone for original text dataframe
tone_lmdict = [tone_count_with_negation_check(sentiment_dict, x.lower()) for x in train_df['sentences']]

tone_lmdict_df = pd.DataFrame(tone_lmdict, columns=columns)
train_df = pd.concat([train_df, tone_lmdict_df.reindex(train_df.index)], axis=1)

# Show corelations to next_decision
plt.figure(figsize=(10,6))
corr_columns = ['polarity', 'labels', 'tone_score', 'word_count', 'n_pos_words', 'n_neg_words']
sns.heatmap(train_df[corr_columns].astype(float).corr(), cmap="coolwarm", annot=True, fmt=".2f", vmin=-1, vmax=1)
# plt.show(block = True);

# X and Y data used
train_df.dropna(inplace = True)
Y_data = train_df['labels']
X_data = train_df[['polarity','tone_score', 'n_pos_words', 'n_neg_words']]


opt2 = input("Choose a classifier : "
                   "\n\n\t 'lr' to select logistic regression" 
                   "\n\t 'ls' to select Linear SVC" 
                   "\n\t 's' to select SVM" 
                   "\n\t 'dt' to select Decision Tree"   
                   "\n\t 'dt' to select Random Forest"
                   "\n\t 'mn' to select multinomial naive bayes \n\n")

trn_data, tst_data, trn_cat, tst_cat = train_test_split(X_data, Y_data, test_size=0.20, random_state=42,stratify=Y_data)   
#     print(trn_data)
#     print(trn_cat)


# Naive Bayes Classifier
if opt2=='mn':      
    clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
    clf_parameters = {
    'clf__alpha':(0,1),
    }  
# SVM Classifier
elif opt2=='ls': 
    clf = svm.LinearSVC(class_weight='balanced')  
    clf_parameters = {
    'clf__C':(0.1,1,2,10,50,100),
    }   
elif opt2=='s':
    clf = svm.SVC(kernel='linear', class_weight='balanced')  
    clf_parameters = {
    'clf_kernel':('poly','linear','sigmoid'),
    'clf__C':(0.1,0.5,1,2,10,50,100)
    }   
# Logistic Regression Classifier    
elif opt2=='lr':    
    clf=LogisticRegression(penalty='l2', class_weight='balanced') 
    clf_parameters = {
    'clf__solver':('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
    }    
# Decision Tree Classifier
elif opt2=='dt':
    clf = DecisionTreeClassifier(class_weight='balanced', random_state=40)
    clf_parameters = {
    'clf__criterion':('gini', 'entropy'), 
    'clf__max_depth': (10, 20, 30, 50),
    'clf__max_features':('auto', 'sqrt', 'log2'),
    'clf__ccp_alpha':(0.01,0.02,0.03,0.05,0.08,0.1)
    }  
# Random Forest Classifier    
elif opt2=='rf':
    clf = RandomForestClassifier(class_weight='balanced')
    clf_parameters = {
    'clf__criterion':('gini', 'entropy'), 
    'clf__max_features':('auto', 'sqrt', 'log2'),   
    'clf__n_estimators':(30,50,100,200),
    'clf__max_depth':(10, 20, 30, 50),
    }     

elif opt2 == 'xgb':
    clf = XGBClassifier()
    clf_parameters = {
    'clf__learning_rate': [0.05, 0.1, 0.2],
    'clf__max_depth': [7, 10, 15],
    'clf__min_child_weight':[0.5, 1, 3, 5],
    'clf__subsample': [0.8, 0.7],
    'clf__n_estimators' : [10, 50, 200]
    }

else:
    print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
    sys.exit(0)                                  

# Feature Extraction
pipeline = Pipeline([
('clf', clf)
]) 

# Classificaion
parameters={**clf_parameters} 
grid = GridSearchCV(pipeline,parameters,scoring='f1_macro',cv=10)          
grid.fit(trn_data,trn_cat)     
clf= grid.best_estimator_  
print('********* Best Set of Parameters ********* \n\n')
print(grid.best_params_)
#     print(clf)

predicted = clf.predict(tst_data)
predicted = list(predicted)

# Evaluation
print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
print('\n Total documents in the test set: '+str(len(tst_data))+'\n')
print ('\n Confusion Matrix \n')  
print (confusion_matrix(tst_cat, predicted))  

print('Classification Report\n',classification_report(tst_cat, predicted))

pr=precision_score(tst_cat, predicted, average='macro') 
print ('\n Precision:'+str(pr)) 

rl=recall_score(tst_cat, predicted, average='macro') 
print ('\n Recall:'+str(rl))

fm=f1_score(tst_cat, predicted, average='macro') 
print ('\n Macro Averaged F1-Score:'+str(fm))
