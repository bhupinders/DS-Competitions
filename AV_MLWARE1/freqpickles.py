import pandas as pd
import numpy as np
from collections import Counter
from nltk import PorterStemmer

train = pd.read_csv('train_MLWARE1.csv')

sarcasm = train.loc[train['label']=='sarcastic']
nosarcasm = train.loc[train['label']=='non-sarcastic']

def get_freqency_list_of_words(list_of_tweets, breakpoint):
    freq_list = Counter(" ".join(list_of_tweets).split(" "))
    freq_listsorted = freq_list.most_common()
    index_of_break = 0
    for i in range(len(freq_listsorted)):
        #print(freq_listsorted[i][1])
        if freq_listsorted[i][1] < breakpoint:
            index_of_break = i
            break
    hf = freq_listsorted[:index_of_break]
    lf = freq_listsorted[index_of_break:]
    return [i[0] for i in hf],[i[0] for i in lf]


def get_processed_words_list(words_list):
    wordstoRemove = ('and','or','of','the','a','to','is','for','in','on','at','you','from',
                 'this','', 'it','','an','has','are','as','by','can','with','that','your',
                 'will','also','well','any','see','only','its','set','any','few','now','but','all')

    import re
    for i in range(len(words_list)):
        words_list[i] = re.sub('[^A-Za-z]+', '',words_list[i]).lower()

    for i in words_list[:]:
        try :
            a =  str(i)
        except :
            a = ' '
        if a in wordstoRemove:
            words_list.remove(i)

        elif len(i) < 3:
            words_list.remove(i)

    return words_list

sarcasm = train.loc[train['label']=='sarcastic']
nosarcasm = train.loc[train['label']=='non-sarcastic']

print("Getting Freqency List")
hf_sar, lf_sar = get_freqency_list_of_words(sarcasm.tweet.values, 300)
hf_no_sar, lf_no_sar = get_freqency_list_of_words(nosarcasm.tweet.values, 300)

hf_sar = get_processed_words_list(hf_sar)
lf_sar = get_processed_words_list(lf_sar)
hf_no_sar = get_processed_words_list(hf_no_sar)
lf_no_sar = get_processed_words_list(lf_no_sar)

import pickle
print("Dumping")
pickle.dump(hf_sar, open('hf_sar.p', 'wb'))
pickle.dump(lf_sar, open('lf_sar.p', 'wb'))
pickle.dump(hf_no_sar, open('hf_no_sar.p', 'wb'))
pickle.dump(lf_no_sar, open('lf_no_sar.p', 'wb'))

hf, lf = get_freqency_list_of_words(train.tweet.values, 1000)

hf = get_processed_words_list(hf)
pickle.dump(hf, open('hf.p', 'wb'))
