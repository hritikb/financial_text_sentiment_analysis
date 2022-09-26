import csv,sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV 
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
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest,chi2 
from preprocess import data


def Build_Model(dataset, labels, opt2):

    # Training and Test Split           
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(dataset, labels, test_size=0.20, random_state=42,stratify=labels)   
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
    ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
    ('feature_selector', SelectKBest(chi2, k=1000)),         
    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),
    ('clf', clf)
    ]) 
        
    feature_parameters = {
    'vect__min_df': (2,3),
    'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3))  # Unigrams, Bigrams or Trigrams
    }
    
# Classificaion
    parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_macro',cv=10)          
    grid.fit(trn_data,trn_cat)     
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(grid.best_params)
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
    
    return clf

