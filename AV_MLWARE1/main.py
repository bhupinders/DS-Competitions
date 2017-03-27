import pandas as pd
from sklearn import cross_validation
import numpy as np
import pickle
from newfeatures import *
import re

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train_MLWARE1.csv')
test = pd.read_csv('test_MLWARE1.csv')

#tweet = 'Hi! there, wassup #cool #hashtags #LOL'
#print(preprocess(tweet))

process(train)
#new_cols_names = add_words_vector(train)

train.loc[train['label']=='sarcastic','label'] = 1
train.loc[train['label']=='non-sarcastic','label'] = 0

cols = ['sarcasm_counter','single_quote_counter','double_quote_counter','repetition',
        'seriously_hashtag','fun_words','all_capital_counter','first_capital_counter',
        'not_hashtag','tweet','hf_sar','lf_sar','hf_no_sar','lf_no_sar','lexical_density','no_vowel_words','label']

#cols = cols + new_cols_names

trainRandom = train.iloc[np.random.permutation(len(train))]

X = trainRandom[cols]
y = list(trainRandom.label.values)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)


model = RandomForestClassifier(n_estimators = 100, random_state = 42, max_depth = 10)
print("Training model....")
model.fit(X_train.drop(['tweet','label'],axis = 1), y_train)
print("Training done")

y_pred = model.predict(X_test.drop(['tweet','label'],axis = 1))

print("F - Score : ", f1_score(y_test,y_pred))

print("Predicting....")
process(test)
#new_cols_names = add_words_vector(test)

pred_cols = [c for c in cols if c != 'label']
X_pred = test[pred_cols]
y_ans = model.predict(X_pred.drop('tweet',axis = 1))
print("Done!!")

## Save answers
answers = pd.DataFrame({'ID':test.ID,
                         'label':y_ans})
answers.loc[answers['label']==1,'label'] = 'sarcastic'
answers.loc[answers['label']==0,'label'] = 'non-sarcastic'

answers.to_csv('file_name.csv', index = False)
