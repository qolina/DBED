#!/usr/bin/python
# -*- coding: UTF-8 -*-

## function
## test the performance of vector

import sys
import os
import re
import time
from pprint import pprint
from collections import Counter
import numpy as np

from sklearn.metrics import pairwise
from gensim import corpora, models, similarities

def testVec_byNN_gensim(doc2vecModel, texts):
    ###############################
    # how many tweets are semantically similar to given tweet
    simThreshold = 0.95
    simFreqCount = []
    simNN = []
    for docid, docvec in enumerate(doc2vecModel.docvecs):
        nn = doc2vecModel.docvecs.most_similar(docid, topn=10000)
        nn = [item for item in nn if item[1] > simThreshold]
        #nn = [item for item in enumerate(nn) if item[1] > simThreshold]
        simFreqCount.append(len(nn))
        simNN.append(nn)

    # sort tweet by simFreq
    sortedTweet_byCount = sorted(enumerate(simFreqCount), key = lambda item: -item[1])
    freqCounts = [item[1] for item in sortedTweet_byCount]
    print Counter(freqCounts).most_common()
    for docid, freqCount in sortedTweet_byCount[:]:
        if "apple" not in texts[docid]:
            continue
        if freqCount == 0:
            break
        print "############################"
        print "**", docid, freqCount
        print "**", texts[docid]
        for item in simNN[docid]:
            print item[0], item[1], " ".join(texts[int(item[0][5:])])

def testVec_byNN(nns_fromSim, texts, Knn):
    for docid in [0, 1, 2]:
        nn = nns_fromSim[docid]
        nn_texts = [texts[idx[0]] for idx in nn]
        print "Tweet", docid, len(nn), "\t", texts[docid]
        outputNN(nn, nn_texts)

def outputNN(nn, texts):
    print "############################"
    for itemIdx in range(len(nn))[:10]:
        idx, sim = nn[itemIdx]
        if sim is None or sim >= 0.5:
            print idx, sim, texts[itemIdx]

def testVec_byLSH(ngIdxArray, texts):
    for docid in [0, 1, 2]:
        nn = [(item, None) for item in ngIdxArray[docid]]
        nn_texts = [texts[idx[0]] for idx in nn]
        print "Tweet", docid, len(nn), "\t", texts[docid]
        outputNN(nn, nn_texts)
        
