import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize, TreebankWordTokenizer, TweetTokenizer, MWETokenizer
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string

import numpy as np
import pandas as pd

from collections import Counter
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import seaborn as sns

train_corpus = pd.read_csv('project2/project2_training_data.txt', delimiter='\n', header=None,names=['sentences'])
train_labels = pd.read_csv('project2/project2_training_data_labels.txt', delimiter='\n', header=None, names=['labels'])

df = pd.concat([train_corpus, train_labels], axis=1)

pd.set_option('max_colwidth', 600)

# Corpus
def create_corpus(df):
    corpus = []
    
    for x in df['cleaned'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


# Returns Top X frequent non stop words
def get_frequent_words(corpus, top_n=10):
    dic = dict()
    for word in corpus:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1

    top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return zip(*top)

# Remove duplicates (3 in the given dataset)
dups = df[df.duplicated()]
df.drop_duplicates(inplace=True)

# Remove stopwords
stops = set(stopwords.words('english'))

# Keep a few relevant stopwords
stopwords = list(set(stopwords.words("english")) - {"aren't", 'above', 'couldn', "couldn't", 'didn', "didn't",'doesn',"doesn't", 'don',"don't", 'below', 'before', 'down','hadn',
                                               "hadn't",'hasn',"hasn't", 'haven',"haven't","isn't", 'isn', 'mightn',"mightn't", 'mustn', "mustn't", 'needn', "needn't", 'more', 
                                               'further','from','no','nor','not', 'over', 'shan',"shan't", 'shouldn',"shouldn't", 'to','under', 'up','wasn',"wasn't",'weren',"weren't",
                                               'won',"won't",'wouldn', "wouldn't",})

# Remove punctuations, make lowercase
def clean_data(text, stopwords):
    tokens = word_tokenize(text.strip())
        
    lower = [i.lower() for i in tokens]
    
    clean = [j for j in lower if j not in stopwords]
    
    punctuations = list(string.punctuation) + ['+', '-', '*', '/']
    clean = [k.strip(''.join(punctuations)) for k in clean if k not in punctuations]
    if 's' in clean:
        clean.remove('s')
    return ' '.join(clean)

df['cleaned'] = df['sentences'].apply(lambda x: clean_data(x, stopwords))

labels = df.labels.values
data =df['cleaned'].values

# Create corpus of our data
corpus = create_corpus(df)

# Check most frequent words which are not in stopwords
counter = Counter(corpus)
most = counter.most_common()[:60]
x, y = [], []
for word, count in most:
    x.append(word)
    y.append(count)

# Plot the Word Cloud
plt.figure(figsize=(15,15))
sns.barplot(x=y, y=x);



choice = input('Do you want to look at the word clouds for the different classes?(y/n)\n')
if choice.lower() == 'y':
    # Make Wordclouds

    # All words
    comment_words = ''

    # iterate through the csv file
    for val in df.cleaned:
        
        # split the value
        tokens = val.split()
        
        comment_words += " ".join(tokens)+" "
        
    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    min_font_size = 10).generate(comment_words)
    
    # plot the WordCloud image                      
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.title('WordCloud of all words')
    plt.axis("off")
    plt.tight_layout(pad = 0)
    
    # plt.show()

    # For Positive & Negative words
    pos_words = ''
    neg_words = ''

    for sentiment in ['positive', 'negative', 'neutral']:
        # iterate through the csv file
        for val in df[df.labels == sentiment].cleaned:
            
            # split the value
            tokens = val.split()
            
            pos_words += " ".join(tokens)+" "

        wordcloud = WordCloud(width = 800, height = 800,
                        background_color ='white',
                        min_font_size = 10).generate(pos_words)
        
        # plot the WordCloud image                      
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.title(f'Word Cloud of {sentiment}ly classed texts\n\n', fontsize = 20)
        
        # plt.show();

else:
    pass

num_words_per_sentence = df['cleaned'].apply(lambda x: len(nltk.word_tokenize(x)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))


sns.countplot(x='labels', data=df, ax=ax1)
ax1.set_title('Sentiment Distribution', fontsize=16)
ax2.hist(num_words_per_sentence,bins = 16)
ax2.set_xlabel('The number of words per sentence')
ax2.set_title('Words Distribution', fontsize=16);

print("Pre-processing Done!")
