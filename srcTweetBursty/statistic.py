
from tweetSim import testVec_byNN
from collections import Counter
from sklearn.metrics import pairwise
import numpy as np

def idxTimeWin(dayTweetNumHash, timeWindow):
    test = False
    if dayTweetNumHash is None:
        test = True
        dayTweetNumArr = [i for i in range(1, 10)]
    else:
        dayTweetNumArr = sorted(dayTweetNumHash.items(), key = lambda a:a[0])
        dayTweetNumArr = [item[1] for item in dayTweetNumArr]
    numSum = [sum(dayTweetNumArr[:i+1]) for i in range(len(dayTweetNumArr))]
    numSumPre = [sum(dayTweetNumArr[:i]) for i in range(len(dayTweetNumArr))]
    dayWindow = []
    dayRelWindow = []
    for date in range(len(dayTweetNumArr)):
        tw1stDate = max(0, date + timeWindow[0])
        twLstDate = min(date + timeWindow[1], len(dayTweetNumArr)-1)
        start = numSumPre[tw1stDate]
        end = numSum[twLstDate]
        dayWindow.append((start, end))
        rel_start = numSumPre[date] - numSumPre[tw1stDate]
        rel_end = rel_start + dayTweetNumArr[date]
        dayRelWindow.append((rel_start, rel_end))
    if test:
        print "## Tesing index extraction in time window"
        print dayTweetNumArr
        print numSum
        print numSumPre
        print dayWindow
        print dayRelWindow
        print "## End Tesing index extraction in time window"
    return dayWindow, dayRelWindow

def getValidDayWind(validDays, dayArr, dayWindow, dayRelWindow):
    validDayWind = [] #[None]*len(dayWindow)
    for day in dayArr:
        (st, end) = dayWindow[int(day)-1]
        (rel_st, rel_end) = dayRelWindow[int(day)-1]
        if day in validDays:
            validDayWind.append((st, end))
        else:
            validDayWind.append((rel_end-rel_st,))
    return validDayWind


def distDistribution(dataset):
    #dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
    distmatrix = pairwise.euclidean_distances(dataset, dataset)
    #distmatrix = pairwise.cosine_distances(dataset, dataset)
    ds = distmatrix.flatten()
    ds = [round(item, 1) for item in ds]

    ds = Counter(ds)
    num = sum([item[1] for item in ds.items()])
    print [(item[0],round(float(item[1])/num, 3)) for item in sorted(ds.items(), key = lambda a:a[0])]
    print sorted(ds.items(), key = lambda a:a[0])
    print "Distance distri cal ended.", time.asctime()

def zsDistribution(zscoresStat):
    tNum = len(zscoresStat)
    print "## statistic of zscore in #tweet", tNum, sum(zscoresStat)/tNum#, Counter(zscoresStat).most_common()
    zscoresStat = Counter(zscoresStat)
    print "## #zs, max, min, avg", len(zscoresStat.keys()), max(zscoresStat.keys()), min(zscoresStat.keys())
    sorted_zsStat = sorted(zscoresStat.items(), key = lambda a:a[0])
    sorted_zsStat = [round(item[1]*100.0/tNum, 2) for item in sorted_zsStat]
    cumu_zsStat = [round(sum(sorted_zsStat[:idx+1]), 2) for idx, num in enumerate(sorted_zsStat)]
    for zs, num in zip(sorted(zscoresStat.keys()), sorted_zsStat):
        print zs, "\t", num
    print zip(sorted(zscoresStat.keys()), sorted_zsStat)
    print zip(sorted(zscoresStat.keys()), cumu_zsStat)
    print sorted_zsStat
    print cumu_zsStat


def statGoldNews(dayNews):
    for item in dayNews[0]:
        print item[1]
    print [(did, len(dayNews_day)) for did, dayNews_day in enumerate(dayNews)]
    print sum([len(dayNews_day) for dayNews_day in dayNews[1:]])


# statistic wordnum
def stat_wordNum(tweetTexts_all, seqDayHash, validDays):
    tweets = []
    for docid in seqDayHash:
        if seqDayHash[docid] in validDays:
            tweets.append(tweetTexts_all[docid].lower())
    wordsDict = Counter(" ".join(tweets).split())
    cashtag = [num for word,num in wordsDict.items() if word[0] == "$" and len(word) > 1 and word[1].isalpha()==True]
    print "Stat: tNum", len(tweets)
    print "Stat: wNum", len(wordsDict)
    print "Stat: cashtag", len(cashtag), sum(cashtag)*1.0/len(tweets)

# testing vec's performace by finding nearest neighbor
def stat_nn_performance(dataset, tweetTexts_all):
    dataset = dataset[:1000, :]
    simMatrix = pairwise.cosine_similarity(dataset)
    nns_fromSim = [sorted(enumerate(simMatrix[i]), key = lambda a:a[1], reverse=True)[:100] for i in range(simMatrix.shape[0])]
    print "## Similarity Matrix obtained at", time.asctime()
    testVec_byNN(nns_fromSim, tweetTexts_all, 10)


def output_zsDistri_day(validDays, zscoreDayArr, simDfDayArr, dayTweetNumHash, tweetTexts_all):
    for pDate in validDays:
        print "###############################################"
        #if pDate != "18": continue
        zs_pDate = zscoreDayArr[int(pDate)-1]
        df_pDate = simDfDayArr[int(pDate)-1]
        startNum = sum([num for dayItem, num in dayTweetNumHash.items() if int(dayItem)<int(pDate)])
        zs_pDate = [(docid, zscoreday[0][1]) for docid, zscoreday in enumerate(zs_pDate)]

        zs_distri = Counter([round(item[1], 0) for item in zs_pDate])
        print "## zs valid distri in days", sorted(zs_distri.items(), key = lambda a:a[0])
        print len(tweetTexts_all)

        texts = [tweetTexts_all[docid+startNum] for docid in range(len(zs_pDate))]
        print texts[0]
        print texts[109]
        print texts[-1]
        textCounter = Counter(texts)
        zs_text = [texts.index(text) for text, num in textCounter.items()]
        print "## Statistic zs in day", pDate, len(zs_pDate), " #uniqText", len(textCounter)

        sorted_zs = sorted(zs_pDate, key = lambda a:a[1], reverse=True)#[:1000]
        clickTweet = []
        for sortedId, twItem in enumerate(sorted_zs):
            docid, zs = twItem
            if docid not in zs_text: continue
            text = texts[docid]
            appnum = textCounter[text]
            if text.lower().find("click here") >= 0:
                clickTweet.append(sortedId)
            if sortedId < 1000:
                print docid, appnum , zs, sorted(df_pDate[docid].items(), key = lambda a:a[0]), text
        print "** ClickHere Tweet num", len(clickTweet), "ratioInAll", len(clickTweet)*100.0/len(sorted_zs), " avgLocation", np.mean(clickTweet)
        #for docid, zs in sorted_zs[-20:]:
        #    print docid, zs, sorted(df_pDate[docid].items(), key= lambda a:a[0]), tweetTexts_all[docid+startNum]

def output_zsDistri_all_tbd():
    #clusters = [item[0] for item in clusters]
    for pDate in dayArr:
        zs_pDate = [(docid, zscoreDay[0][1]) for docid, zscoreDay in enumerate(zscoreDayArr) if zscoreDay is not None if seqDayHash[docid] == pDate]
        uniq_docs = zs_pDate
        #uniq_docs = [(docid, zs) for docid, zs in zs_pDate if docid in clusters]
        print "## Statistic zs in day", pDate, "unique/all", len(uniq_docs), len(zs_pDate)
        if len(uniq_docs) < 10: continue
        #texts = [tweetTexts_all[docid] for docid, zs in uniq_docs]
        #dataset = getVec('3', None, None, None, word2vecModelPath, texts)
        sorted_zs = sorted(uniq_docs, key = lambda a:a[1], reverse=True)
        for docid, zs in sorted_zs[:50]:
            print docid, zs, sorted(simDfDayArr[docid].items(), key = lambda a:a[0]), tweetTexts_all[docid]
        for docid, zs in sorted_zs[-50:]:
            print docid, zs, sorted(simDfDayArr[docid].items(), key = lambda a:a[0]), tweetTexts_all[docid]

def stat_snpRatio_tbd():
    w2v = loadWord2Vec(word2vecModelPath)
    cashtag = [item for item in w2v if item[0]=="$"]
    snpRatio = [item for item in cashtag if item[1:] in snp_syms]
    print len(snpRatio)
    print "\n".join(sorted(snpRatio))

# output tfidf vec file by day for large data set nn calculate
# may not be used anymore
def output_tfidfvec_day_tbd():
    for i, vdw in enumerate(validDayWind):
        if len(vdw) == 1: continue
        dataIdx_v = range(vdw[0], vdw[1])
        #tweets.extend([tweetTexts_all[docid] for docid in dataIdx_v])

        dataset_v = dataset[dataIdx_v,:]
        dataFile = open("../ni_data/tweetVec/tweets.tfidf."+str(i), "w")
        cPickle.dump(dataset_v, dataFile)
        dataFile.close()


