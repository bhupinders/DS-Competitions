import pandas as pd
from sklearn import cross_validation
import numpy as np
import pickle
import re

## Copied from some blog
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def get_array_from_tweet(tweet):
    return preprocess(tweet)
##

#### New Features
def find_sarcasm_words(tweet):
    #text = str(row).lower()
    match = 'sarcas'
    #find = len(text.split(match)) - 1
    tweetarr = get_array_from_tweet(tweet)
    counter = 0
    for s in tweetarr[:]:
        if match in s.lower():
            counter = counter + 1
    return counter

def find_exclamation_word(tweet):
    match = '!'
    #find = len(text.split(match)) - 1
    tweetarr = get_array_from_tweet(tweet)
    counter = 0
    for s in tweetarr[:]:
        if match in s:
            counter = counter + 1
    return counter

def find_single_quote_word(tweet):
    match = "'"
    #find = len(text.split(match)) - 1
    tweetarr = get_array_from_tweet(tweet)
    counter = 0
    for s in tweetarr[:]:
        if match in s:
            counter = counter + 1
    return counter

def find_double_quote_word(tweet):
    match = '"'
    #find = len(text.split(match)) - 1
    tweetarr = get_array_from_tweet(tweet)
    counter = 0
    for s in tweetarr[:]:
        if match in s:
            counter = counter + 1
    return counter

def find_dots_word(tweet):
    match = '.'
    #find = len(text.split(match)) - 1
    tweetarr = get_array_from_tweet(tweet)
    counter = 0
    for s in tweetarr[:]:
        if match in s:
            counter = counter + 1
    return counter

def find_all_capital_word(tweet):
    tweetarr = get_array_from_tweet(tweet)
    counter = 0
    for s in tweetarr[:]:
        if s.isupper():
            counter = counter + 1
    return counter

def find_first_capital_word(tweet):
    tweetarr = get_array_from_tweet(tweet)
    counter = 0
    for s in tweetarr[:]:
        if s[0].isupper():
            counter = counter + 1
    return counter

def find_tweet_length(tweet):
    return len(get_array_from_tweet(tweet))

def find_not_hashtag(tweet):
    match = '#not'
    tweetarr = get_array_from_tweet(tweet)
    counter = 0
    for s in tweetarr[:]:
        if match == s.lower():
            counter = counter + 1
    return counter

def find_repetition(tweet):
    tweetarr = get_array_from_tweet(tweet)
    repeat = 0
    for i in range(len(tweetarr)-1):
        if tweetarr[i].lower() == tweetarr[i+1].lower():
            repeat = 1
    return repeat

def find_question_mark_word(tweet):
    match = '?'
    #find = len(text.split(match)) - 1
    tweetarr = get_array_from_tweet(tweet)
    counter = 0
    for s in tweetarr[:]:
        if match in s:
            counter = counter + 1
    return counter

def find_seriously_hashtag(tweet):
    match = '#serious'
    tweetarr = get_array_from_tweet(tweet)
    counter = 0
    for s in tweetarr[:]:
        if match in s.lower():
            counter = counter + 1
    return counter

#def find_fun_words(tweet):
#    match = ['fun','wow','laugh','love','awesom','great','haha']
#    tweetarr = get_array_from_tweet(tweet)
#    counter = 0
#    for s in tweetarr[:]:
#        for m in match[:]:
#            if m in s.lower():
#                counter = counter + 1
#    return counter
def find_fun_words(tweet):
    match = 'fun'
    tweetarr = get_array_from_tweet(tweet)
    counter = 0
    for s in tweetarr[:]:
        if match in s.lower():
            counter = counter + 1
    return counter

def find_long_hashtags(tweet):
    match = '#'
    tweetarr = get_array_from_tweet(tweet)
    counter = 0
    for s in tweetarr[:]:
        if match in s:
            if len(s)>10:
                counter = counter + 1
    return counter

def find_frequency(tweet, **wordslisttemp):
    for i in wordslisttemp:
        wordslist = i
    import re
    arr = tweet.split(" ")
    for i in range(len(arr)):
        arr[i] = re.sub('[^A-Za-z]+', '',arr[i]).lower()
    counter = 0
    for s in arr[:]:
        if s in wordslist:
            counter = counter + 1
    return counter

def find_lexical_density(tweet):
    tweetarr = get_array_from_tweet(tweet)

    uniqueWords = []
    for i in tweetarr[:]:
        if not i in uniqueWords:
            uniqueWords.append(i);

    return len(uniqueWords) / len(tweetarr)

def no_vowel_words(tweet):
    tweetarr = get_array_from_tweet(tweet)
    counter = 0
    pattern = '^[^aeyiuo]+$'
    for i in tweetarr[:]:
        if re.search(pattern,i):
            counter = counter+1
    return counter

#### Process Dataframe
def process(df):
    df['sarcasm_counter'] = 0
    df['exclamation_counter'] = 0
    df['single_quote_counter'] = 0
    df['double_quote_counter'] = 0
    df['dots_counter'] = 0
    df['all_capital_counter'] = 0
    df['first_capital_counter'] = 0
    df['tweet_len'] = 0
    df['not_hashtag'] = 0
    df['repetition'] = 0
    df['question_mark_counter'] = 0
    df['seriously_hashtag'] = 0
    df['fun_words'] = 0
    df['long_hashtags'] = 0

    df['hf_sar'] = 0
    df['lf_sar'] = 0
    df['hf_no_sar'] = 0
    df['lf_no_sar'] = 0

    df['lexical_density'] = 0
    df['no_vowel_words'] = 0

    df['sarcasm_counter'] = df['tweet'].apply(find_sarcasm_words)

    df['exclamation_counter'] = df['tweet'].apply(find_exclamation_word)

    df['single_quote_counter'] = df['tweet'].apply(find_single_quote_word)

    df['double_quote_counter'] = df['tweet'].apply(find_double_quote_word)

    df['dots_counter'] = df['tweet'].apply(find_dots_word)

    df['all_capital_counter'] = df['tweet'].apply(find_all_capital_word)

    df['first_capital_counter'] = df['tweet'].apply(find_first_capital_word)

    df['tweet_len'] = df['tweet'].apply(find_tweet_length)

    df['not_hashtag'] = df['tweet'].apply(find_not_hashtag)

    df['repetition'] = df['tweet'].apply(find_repetition)

    df['question_mark_counter'] = df['tweet'].apply(find_question_mark_word)

    df['seriously_hashtag'] = df['tweet'].apply(find_seriously_hashtag)

    df['fun_words'] = df['tweet'].apply(find_fun_words)

    df['long_hashtags'] = df['tweet'].apply(find_long_hashtags)

    df['lexical_density'] = df['tweet'].apply(find_lexical_density)

    df['no_vowel_words'] = df['tweet'].apply(no_vowel_words)

    ## Frequency
    hf_sar = pickle.load( open( "hf_sar.p", "rb" ) )
    lf_sar = pickle.load( open( "lf_sar.p", "rb" ) )
    hf_no_sar = pickle.load( open( "hf_no_sar.p", "rb" ) )
    lf_no_sar = pickle.load( open( "lf_no_sar.p", "rb" ) )

    df['hf_sar'] = df['tweet'].apply(find_frequency, ls = hf_sar)

    df['lf_sar'] = df['tweet'].apply(find_frequency, ls = lf_sar)

    df['hf_no_sar'] = df['tweet'].apply(find_frequency, ls = hf_no_sar)

    df['lf_no_sar'] = df['tweet'].apply(find_frequency, ls = lf_no_sar)

## Words Vector
hf = pickle.load( open( "hf.p", "rb" ) )

def get_feature_vector(tweet):
    vector = np.zeros(len(hf))
    arr = tweet.split(" ")
    for i in range(len(arr)):
        arr[i] = re.sub('[^A-Za-z]+', '',arr[i]).lower()
    for i in range(len(arr)):
        if arr[i] in hf:
            index = hf.index(arr[i])
            vector[index] = 1

    return vector

def add_words_vector(df):
    print("Adding words Vector....")
    newcols = []
    for i in range(len(hf)):
        newcols.append('words_feature_'+str((i+1)))
        df['words_feature_'+str((i+1))] = 0

    #df[newcols] = df['tweet'].apply(get_feature_vector)
    for i in range(df.shape[0]):
        if(i%5000==0):
            print('Done : ', i)
        df.loc[i,newcols] = get_feature_vector(df.loc[i,'tweet'])
    print("Done")
    return newcols
