import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize

useless_punc = ['-','/']

def extract_tweets():
    tweets = []
    data = pd.read_csv('trumptweets.csv').dropna(subset=['text'])
    tweets = data['text'].tolist()

    return tweets

def data_clean(tweets):
    cleaned_tweets = []

    for tweet in tweets:

        new_tweet = tweet.lower()
        new_tweet = re.sub(r'http(.*?)((\s)+|$)','',new_tweet)
        new_tweet = re.sub(r'\.+','.',new_tweet)
        new_tweet = new_tweet.replace('&amp','and')
        tokens = word_tokenize(new_tweet)
        new_tokens = []
        #Removes @s
        for i in range(len(tokens)):
            if (tokens[i]=='@'):
                continue
            elif (tokens[i-1]=='@'):
                continue
            elif (tokens[i] in useless_punc):
                continue
            else:
                new_tokens.append(tokens[i])

        new_tweet = ' '.join(tokens)
        cleaned_tweets.append(new_tweet)

    return cleaned_tweets

def data_preprocess(tweets,THRES_FREQ):
    vocab = dict()
    all_tokens = []
    for tweet in tweets:
        tokens = word_tokenize(tweet)
        for token in tokens:
            if token not in vocab:
                vocab[token]=1
            else:
                vocab[token]+=1

    vocab_freq = sorted([(k,v) for k,v in vocab.items()],reverse=True,key = lambda x:x[1])
    vocab_freq = [x for x in vocab_freq if (x[1] >= THRES_FREQ) ]
    vocab = dict(vocab_freq)

    for tweet in tweets:
        tokens = word_tokenize(tweet)
        tokens = [token for token in tokens if token in vocab]
        all_tokens.append(tokens)

    return all_tokens

def main():
    tweets = extract_tweets()
    tweets = data_clean(tweets)
    tweets = data_preprocess(tweets,10)

    tweets = np.array(tweets)
    np.save('tweets.npy',tweets)

if __name__ == "__main__":
    main()
