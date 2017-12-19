import os
import sys
import time
import timeit
import math
from collections import Counter
from statistic import zsDistribution

import numpy as np

def getDF(date, ngIdxArray, seqDayHash, timeWindow):
    simDfDayArr = []
    for docid, nnIdxs in enumerate(ngIdxArray):
        if nnIdxs is None:
            simDfDayArr.append(None)
            continue
        nnDays = [seqDayHash.get(seqid) for seqid in nnIdxs.tolist()]
        nnDay_count = None
        if nnDays is not None:
            nnDay_count = Counter(nnDays)
        simDfDayArr.append(nnDay_count)
    return simDfDayArr

# zscoreDayArr: [zscoreDay_seqid0, seq1, ...]
# zscoreDay_seqid: [(day, zscore), (day, zscore)]
def getBursty(simDfDayArr, dayTweetNumHash, tDate, timeWindow):
    if timeWindow is not None:
        tw = [str(int(tDate)+i).zfill(2) for i in range(timeWindow[0], timeWindow[1]+1)]

    zscoreArr = []
    for docid, nnDayCounter in enumerate(simDfDayArr):
        statArr = []

        if timeWindow is not None:
            nnDayCounter = dict([(d, nnDayCounter[d]) for d in tw if nnDayCounter[d] > 0])
            TweetNum_all = sum([n for d, n in dayTweetNumHash.items() if d in nnDayCounter])

        simDf = nnDayCounter.get(tDate)
        if simDf is None or simDf < 1: 
            zscoreArr.append(-999)
            continue
        TweetNum_day = dayTweetNumHash[tDate]

        if 1:
            dfs = np.asarray(nnDayCounter.values(), dtype=np.float32)
            mu = np.mean(dfs)
            sigma = np.std(dfs)
            #print docid, simDf, mu, sigma
            if len(nnDayCounter) == 1: zscore = 99.0
            else:
                if sigma == 0.0:
                    zscore = 0.0
                else:
                    zscore = round((simDf-mu)/sigma, 4)
            statArr.extend([simDf, mu, sigma, zscore])
        else:
            docSimDF_all = sum(nnDayCounter.values())
            est_prob = docSimDF_all*1.0/TweetNum_all
            mu = est_prob * TweetNum_day
            sigma = math.sqrt(mu*(1-est_prob))
            zscore = round((simDf*1.0-mu)/sigma, 4)

            statArr.extend([simDf, mu, sigma, zscore])
            statArr.extend([est_prob, docSimDF_all, dayTweetNumHash[tDate], TweetNum_all])

        #print docid, day, simDf, mu, est_prob, sigma
        if 1 and tDate in ["14", "15"]:
            print "#################################"
            print sorted(nnDayCounter.items(), key = lambda a:a[0])
            #print statArr
            if statArr[-1] > 2.0:
                print "--df, mu, sigma, zs", statArr
        zscoreArr.append(zscore)
    print "## Cluster zscore [li zscore] obtained at", time.asctime()
    return zscoreArr

