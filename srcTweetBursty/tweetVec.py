#!/usr/bin/python
# -*- coding: UTF-8 -*-

## function
## convert tweet in format of tweetid[\t]tweetText into denseVector
## by doc2vec, lsi, tfidf etc

import sys
import os
import re
import time
from pprint import pprint
from collections import defaultdict
from collections import Counter
import cPickle

import numpy as np
from sklearn import cluster

from gensim import corpora, models, similarities
from gensim.models import TfidfModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from util.fileOperation import loadStopword

from util import fileReader

from word2vec import *

####################################################
def normWordInTweet(word):
    if word[0] == "@":
        word = "<username>"
    elif word.startswith("http"):
        word = "<url>"
    #if word not in word2vecModel:
    #    word = "<unk>"
    return word

def normPuncInTweet(tweet):
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        tweet = tweet.replace(char, ' ' + char + ' ')
    return tweet

def raw2Texts(rawTweets, rmStop, rmMinFreq, minFreq):
    # remove stopwords and tokenize
    stopWords = {}
    if rmStop:
        stopWords = loadStopword("../data/stoplist.dft")
        stopWords["<username>"] = 1
        stopWords["<url>"] = 1
    texts = [normPuncInTweet(doc) for doc in rawTweets]
    texts = [[normWordInTweet(word) for word in document.lower().split() if word not in stopWords] for document in rawTweets]

    if not rmMinFreq:
        return texts

    # remove words that appear only minFreq
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > minFreq] for text in texts]
    return texts

def prepCorpus(texts, dictPath, corpusPath):
    dictionary = corpora.Dictionary(texts)
    dictionary.save(dictPath)

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(corpusPath, corpus)
    #print(corpus[:5])

def getNearestNeighbor(index, simThreshold):
    neighbors = []
    for docSims in index:
        sortedSims = sorted(enumerate(docSims), key=lambda item: -item[1])
        selectedSims = [item for item in sortedSims if item[1] > simThreshold]
        neighbors.append(selectedSims)
        print len(selectedSims), selectedSims[:10]
        break
    return neighbors

def tfidf_vec(dictPath, corpusPath):
    # load dictionary and corpus
    if os.path.exists(dictPath):
        dictionary = corpora.Dictionary.load(dictPath)
        corpus = corpora.MmCorpus(corpusPath)
        print "## corpus loaded."
    else:
        print "## Error: No dictionary and corpus found in ", dictPath
        return -1

    # corpus to model
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    print "## corpus vector end. [tfidf] at ", time.asctime(), len(corpus_tfidf)
    return corpus_tfidf

# lsi example in gensim
def lsi_vec(dictPath, corpusPath, lsiModelPath):
    # load dictionary and corpus
    if os.path.exists(dictPath):
        dictionary = corpora.Dictionary.load(dictPath)
        corpus = corpora.MmCorpus(corpusPath)
        print "## corpus loaded."
    else:
        print "## Error: No dictionary and corpus found in ", dictPath
        return -1

    # corpus to model
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    print "## corpus vector end. [tfidf] at ", time.asctime()
    #lsi = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics = 300)
    #print "## corpus transformation end. [LSI] at ", time.asctime()
    lsi.save(lsiModelPath)

def lsi_vec_sim(dictPath, corpusPath, lsiModelPath, simIndexPath):
    dictionary = corpora.Dictionary.load(dictPath)
    corpus = corpora.MmCorpus(corpusPath)
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # calculate similarity
    lsi = models.LsiModel.load(lsiModelPath)
    corpus_lsi = lsi[corpus_tfidf]
    print corpus_tfidf[0]
    print corpus_lsi[0]
    index = similarities.MatrixSimilarity(corpus_lsi)
    print "## corpus similarity index end. at ", time.asctime()
    index.save(simIndexPath)

    ###############################
    index = similarities.MatrixSimilarity.load(simIndexPath)
    simThreshold = 0.9
    neighbors = getNearestNeighbor(index, simThreshold)

    # example of a doc to whole corpus
    #doc = "This a test of apple watch release"
    #vec_bow = dictionary.doc2bow(doc.lower().split())
    #vec_lsi = lsi[vec_bow]
    #sims = index[vec_lsi]
    #sims = sorted(enumerate(sims), key=lambda item: -item[1])
    #print sims[:10]


def usingDoc2vec(texts, doc2vecModelPath):
    # build doc2vec model and save it
    # do build_vocab and train automatically if initialize documents in building Doc2Vec model
    taggedTexts = [models.doc2vec.TaggedDocument(words=text, tags=["SENT_%s" %tid]) for tid, text in enumerate(texts)]
    doc2vecModel = models.Doc2Vec(taggedTexts, dm=1, dm_mean=0, size=200, window=5, workers=15, iter=20)
    #doc2vecModel = models.Doc2Vec(taggedTexts, dm=0, size=100, negative=5, hs=0, workers=3, iter=20)
    doc2vecModel.save(doc2vecModelPath)


def texts2LSIvecs(texts, dictPath, corpusPath):
    texts = raw2Texts(rawTweetFilename, True, True, 1)
    # prepare corpus from raw
    prepCorpus(texts, dictPath, corpusPath)

    lsi_vec(dictPath, corpusPath)

def texts2TFIDFvecs(texts, dictPath, corpusPath):
    texts = raw2Texts(texts, True, True, 1)
    texts = [" ".join(words) for words in texts]
    count_vect = CountVectorizer()
    texts_counts = count_vect.fit_transform(texts)
    tfidf_transformer = TfidfTransformer(norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    corpus_tfidf = tfidf_transformer.fit_transform(texts_counts)


    # prepare corpus from raw
    #prepCorpus(texts, dictPath, corpusPath)

    #corpus_tfidf = tfidf_vec(dictPath, corpusPath)
    return corpus_tfidf


def texts2vecs(texts, doc2vecModelPath):
    texts = raw2Texts(texts, False, False, None)
    #wordNum = [len(text) for text in texts]
    #print "## min/max/avg words in texts", min(wordNum), max(wordNum), sum(wordNum)*1.0/len(wordNum)
    print "## text leximization finished. ", len(texts), time.asctime()
    usingDoc2vec(texts, doc2vecModelPath)
    print "## doc2vec model saved. ", doc2vecModelPath, time.asctime()


def loadTweetsFromDir(dataDirPath):
    dayTweetNumHash = {} # dayStr:#tweetNum
    tweetTexts_all = []
    seqTidHash = {} # seqId in all: tweet_id
    seqDayHash = {} # seqId in all: dayStr
    for fileItem in sorted(os.listdir(dataDirPath)):
        if not fileItem.startswith("tweetCleanText"):
        #if not fileItem.startswith("tweet"): # for football cup corpus
            continue
        dayStr = fileItem[-2:]
        #dayStr = fileItem[-3:] # for Football Cup corpus
        #if dayStr == "07":break
            
        rawTweetHash = fileReader.loadTweets(dataDirPath + "/" + fileItem) # tid:text
        #print "## End of reading file. [raw tweet file][cleaned text]  #tweets: ", len(rawTweetHash), fileItem
        tids = rawTweetHash.keys()#[:1000]
        texts = rawTweetHash.values()#[:1000]

        word2vecModelPath = "../ni_data/tweetVec/w2v1010100-en"
        #dataset = getVec('3', None, None, len(tweetTexts_all), word2vecModelPath, texts)

        for seqid, tid in enumerate(tids, start=len(tweetTexts_all)):
            seqTidHash[seqid] = tid
            seqDayHash[seqid] = dayStr
        tweetTexts_all.extend(texts)
        
        dayTweetNumHash[dayStr] = len(texts)

    print "## End of reading files. [raw tweet file][cleaned text]  #tweets: ", len(tweetTexts_all), " in days ", len(dayTweetNumHash), time.asctime()
    return tweetTexts_all, seqTidHash, seqDayHash, dayTweetNumHash


##############
# Para_train:   default: '0'
# '0'   : train with finance tweet only. (which is used for clustering)
# '1'   : train with extra large scale finance tweet. (about 2million)
# '2'   : train with extra large scale finance tweet + finance tweet.
def trainDoc2Vec(Para_train, doc2vecModelPath, largeCorpusPath, l_doc2vecModelPath, tweetTexts_all):
    if Para_train == '0':
        texts2vecs(tweetTexts_all, doc2vecModelPath)
    elif Para_train == '1':
        largeCorpus = fileReader.loadTweets(largeCorpusPath).values() # tid:text
        texts2vecs(largeCorpus, l_doc2vecModelPath)
    elif Para_train == '2':
        largeCorpus = fileReader.loadTweets(largeCorpusPath).values() # tid:text
        largeCorpus.extend(tweetTexts_all)
        texts2vecs(largeCorpus, l_doc2vecModelPath)


##############
# para_test:   default: '0'
# '0'   : test with finance tweet trained model only.
# '1'   : test with extra large scale finance tweet trained model.
# '2'   : test with extra large scale + small finance tweet trained model.
# '3'   : test with pretrained word2vec model. (4million words, 100dimension)
def getVec(Para_test, doc2vecModelPath, l_doc2vecModelPath, TweetNum, word2vecModelPath, tweetTexts_all):
    doc2vecModel = None
    dataset = None
    if Para_test == '0':
        doc2vecModel = models.doc2vec.Doc2Vec.load(doc2vecModelPath)
        dataset = np.array(doc2vecModel.docvecs)
        print "## tweet vec by pretrained small doc2vec model obtained.", dataset.shape, time.asctime()
    elif Para_test == '1':
        doc2vecModel = models.doc2vec.Doc2Vec.load(l_doc2vecModelPath)
        # should infer vec in 1
        wordTexts = raw2Texts(tweetTexts_all, False, False, None)
        dataset = []
        for text in wordTexts:
            vec = doc2vecModel.infer_vector(text, steps=5)
            dataset.append(vec)
        dataset = np.asarray(dataset)
        print "## tweet vec by inferring from trained d2v model obtained.", dataset.shape, time.asctime()
    elif Para_test == '2':
        doc2vecModel = models.doc2vec.Doc2Vec.load(l_doc2vecModelPath)
        dataset = np.array(doc2vecModel.docvecs)
        dataset = dataset[-TweetNum:,:]
        print "## tweet vec by pretrained large d2v vec obtained.", dataset.shape, time.asctime()
    elif Para_test == '3':
        word2vecModel = loadWord2Vec(word2vecModelPath)
        print "## word2vec load done.", time.asctime()
        wordTexts = raw2Texts(tweetTexts_all, False, False, None)
        normWordTexts = []
        wordNums = []
        unkNums = []
        for text_seqid, words in enumerate(wordTexts):
            if len(words) == 0: print "--Text seqid in all", text_seqid, "#", tweetTexts_all[text_seqid], "#"
            wordsIn = [word for word in words if word in word2vecModel]
            wordsIn.extend(["<s>", "</s>"])
            unkNum = sum([1 for word in words if word not in word2vecModel])
            wordsIn.extend(["<unk>"]*unkNum)
            normWordTexts.append(wordsIn)
            wordNums.append(len(words))
            unkNums.append(unkNum)
        unkRatio = np.mean([float(un)/wn for un, wn in zip(unkNums, wordNums)])
        dataset = [[np.asarray(word2vecModel.get(word)) for word in words] for words in normWordTexts]
        #dataset = [np.sum(np.asarray(dataset[idx]), axis=0) for idx in range(len(dataset))]
        dataset = [np.mean(np.asarray(dataset[idx]), axis=0) for idx in range(len(dataset))]
        dataset = np.asarray(dataset)
        print "## tweet vec by w2v obtained.", dataset.shape, time.asctime(), "UNK ratio", unkRatio, np.mean(wordNums), np.mean(unkNums)

    return dataset

####################################################
if __name__ == "__main__":
    dataDirPath = parseArgs(sys.argv)
    print "Program starts at ", time.asctime()

    tweetTexts_all, seqTidHash, seqDayHash, dayTweetNumHash = loadTweetsFromDir(dataDirPath)
    #print "\n".join(tweetTexts_all)
    print sorted(dayTweetNumHash.items(), key = lambda a:a[0])

    Para_train, Para_test = ('-', '3')

    doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model"
    l_doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model.large"
    largeCorpusPath = os.path.expanduser("~")+"/corpus/tweet_finance_data/tweetCleanText2016"
    word2vecModelPath = "../ni_data/tweetVec/w2v1010100-en"

    ##############
    # training
    trainDoc2Vec(Para_train, doc2vecModelPath, largeCorpusPath, l_doc2vecModelPath, tweetTexts_all)

    ##############
    # testing/using
    if Para_test in ['0', '1', '2']:
        dataset = getVec(Para_test, doc2vecModelPath, l_doc2vecModelPath, len(tweetTexts_all), word2vecModelPath, None)
    elif Para_test == '3':
        dataset = getVec(Para_test, doc2vecModelPath, l_doc2vecModelPath, len(tweetTexts_all), word2vecModelPath, tweetTexts_all)


    ##############
    # used for lsi, tfidf
    dictPath = "../ni_data/tweetVec/tweets.dict"
    corpusPath = "../ni_data/tweetVec/tweets.mm"
    lsiModelPath = "../ni_data/tweetVec/model.lsi"
    simIndexPath = "../ni_data/tweetVec/tweets.index"

    corpus_tfidf = texts2TFIDFvecs(tweetTexts_all, dictPath, corpusPath)

    #texts2LSIvecs(tweetTexts_all, dictPath, corpusPath)
    #lsi_vec_sim(dictPath, corpusPath, lsiModelPath, simIndexPath)

    print "Program ends at ", time.asctime()
