import os
import sys
import math
import time
import cPickle
from collections import Counter
from sklearn.metrics import pairwise

from tweetVec import * #loadTweetsFromDir, texts2TFIDFvecs, trainDoc2Vec, trainDoc2Vec
from getSim import getSim_falconn, getSim_sparse, getSim_dense
from tweetNNFilt import getDF, getBursty
from tweetClustering import clustering, outputTCluster, compInDoc, clusterTweets, getClusterFeatures, clusterScoring, clusterSummary
from evalRecall import evalTClusters, stockNewsVec, outputEval, dayNewsExtr, evalOutputEvents
from statistic import distDistribution, idxTimeWin, getValidDayWind, statGoldNews, zsDistribution
from statistic import output_zsDistri_day, stat_nn_performance, stat_wordNum

from word2vec import loadWord2Vec

sys.path.append("./util/")
import snpLoader
import stringUtil as strUtil

from args import get_args

seedTweets = ["$71 , 000 in one trade by follwing their signals more info here $cvc $cvd $cve",
"our penny stock pick on $ppch closed up another 51 . 31 today huge news $cce $cadx $grmn",
"our penny stock alerts gained over 3100 in 6 months see our new pick $erx $rig $pot",
"our stock picks have been seeing massive gains this year special update $nov $ery $tza",
"our stock pick on $thcz is up 638 . 15 for our subscribers get our next pick early $tcb $mck $study",
"gains over 2500 in one trade subscribe here $emo $emq $emr",
"largest food and staples retailing earnings 1 $wmt 2 $cvs 3 $wba chart",
"since your tweet was sent $aapl has dropped 3 . 185 see your featured tweet on market parse",
"insider selling silvio barzi sells 5 , 290 shares of mastercard stock $ma",
"volume alert fb 78 . 43 facebook inc $fb hit a high today of 78 . 94 closing the day 05/07/15 at 7    8 . 43 0 . 33 0",

"oracle ceo sees benefit if rival buys salesforce.com",
"dow chemical to sell agrofresh for $860 mln in asset sale drive",
"expedia inc first quarter profit tops expectations",
"conocophillips first quarter profit falls sharply on oil price decline",
"cigna profit beats estimate as it adds more customers",
"solar panel maker first solar reports quarterly loss",
"expedia inc maintains 2015 earnings guidance  cfo",
"mcgraw hill education prepares for ipo",
"obama to push case for trade deal at nike headquarters in oregon",
"tattoo snafu irks inked apple watch wearers"
]

def getClusteringArg(args, tnum):
    clusterArg = None
    if args.cluster == "dbscan":
        clusterArg = args.dbscan_eps 
    else:
        if args.num_cls != -1:
            clusterArg = args.num_cls
        else: # for kmeans: numClusters = [50, 100]  double for hierarchical
            clusterArg = 100
            if tnum >= 20000:
                clusterArg = 200
            if algor == "kmeans":
                clusterArg /= 2
    return clusterArg

# determine some parameters
def params(args, dataSelect=1):
    ######################
    if not args.use_zs: timeWindow = None
    else: timeWindow = (-args.ltw, args.rtw)

    Para_newsDayWindow = [0]
    if args.news_window != 0:
        Para_newsDayWindow = [-1, 0, 1]

    ######################
    validDays = None
    if dataSelect == 1:
        devDays = ['06', '07', '08']
        testDays = ['11', '12', '13', '14', '15']
    elif dataSelect == 2:#['18', '19', '20', '21', '22', '26', '27', '28']
        devDays = ['26', '27', '28']
        testDays = ['18', '19', '20', '21', '22']
    elif dataSelect == 3: #[15-31]
        devDays = ['15', '18', '19']
        testDays = ['20', '21', '22', '26', '27', '28']
    elif dataSelect == 9: #[15-31]
        devDays = [str(i).zfill(2) for i in range(1, 32)]
        testDays = []
        #devDays = ['15']#, '16', '17']
        #validDays_fst = ['06', '07', '08', '11', '12', '13', '14', '15']
        #validDays_weekend = ['09', '10', '16', '17', '23', '24', '25']
        validDays = devDays

    if validDays is None:
        validDays = []
        if args.dev: validDays.extend(devDays)
        if args.test: validDays.extend(testDays)
        validDays = sorted(validDays)

    ######################
    outputDays = validDays
    if args.output == "d": outputDays = devDays
    elif args.output == "t": outputDays = testDays

    return devDays, testDays, validDays, outputDays, timeWindow, Para_newsDayWindow

####################################################
if __name__ == "__main__":
    print "Program starts at ", time.asctime()

    args = get_args()
    print "**Para setting"
    print args

    ##############
    devDays, testDays, validDays, outputDays, timeWindow, Para_newsDayWindow = params(args, dataSelect=1)
    print "validDays", validDays
    print "outputDays", outputDays

    fileSuf_data = os.path.basename(os.path.dirname(args.input+"/")) # eg: "word201505"
    time_flag = "." + time.strftime("%Y%m%d%H%M%S", time.gmtime()) # eg: ".20170912035918"
    output_dir = "../ni_data/models/"+fileSuf_data+"/"
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    if args.cluster == "dbscan": cluster_arg = "eps" + str(args.dbscan_eps)
    else: cluster_arg = "cnum" + str(args.num_cls)

    dfFilePath = output_dir + fileSuf_data + args.df_file + time_flag
    zscoreFilePath = output_dir + fileSuf_data + args.zs_file + "_" + str(args.delta) + time_flag
    clusterFilePath = output_dir + fileSuf_data + args.cls_file + "_" + args.cluster + "_" + cluster_arg + time_flag # eg: '../ni_data/models/word201505/word201505.cluster_20170912035918'
    eventFilePath = output_dir + fileSuf_data + args.evt_file + "_" + args.cluster + "_" + cluster_arg + time_flag # eg: '../ni_data/models/word201505/word201505.cluster_20170912035918'

    ##############
    sym_names = snpLoader.loadSnP500(args.snp_file)
    snp_syms = [snpItem[0] for snpItem in sym_names]
    snp_comp = [strUtil.getMainComp(snpItem[1]) for snpItem in sym_names]
    symCompHash = dict(zip(snp_syms, snp_comp))

    ##############
    # load news vec
    newsVecFile = open(args.news_vectors, "r")
    print "## news obtained for eval", args.news_vectors 
    dayNews = cPickle.load(newsVecFile)
    vecNews = cPickle.load(newsVecFile)
    newsSeqDayHash = cPickle.load(newsVecFile)
    newsSeqComp = cPickle.load(newsVecFile)
    newsVecFile.close()
    #statGoldNews(dayNews)

    ##############
    tweetTexts_all = None
    tweetTexts_all, seqTidHash, seqDayHash, dayTweetNumHash = loadTweetsFromDir(args.input)

    ##############
    dayArr = sorted(dayTweetNumHash.keys()) # ['01', '02', ...]
    if timeWindow is not None:
        dayWindow, dayRelWindow = idxTimeWin(dayTweetNumHash, timeWindow)
        validDayWind = getValidDayWind(validDays, dayArr, dayWindow, dayRelWindow)
        print validDayWind
        print dayRelWindow
    ##############

    ##############
    # testing/using
    if args.do_cls or args.use_zs:
        #tweetTexts_all = tweetTexts_all[:300000]
        dataset = None
        if args.vec == 3 or 3 in args.vec:
            dataset_w2v = getVec('3', None, None, None, args.word_vectors, tweetTexts_all+seedTweets)
            seedTweetVecs = dataset_w2v[range(-20, 0), :]
            if dataset is None:
                dataset = dataset_w2v[:-20,:]
            else:
                # concatenate d2v and w2v
                dataset = np.append(dataset, dataset_w2v, axis=1)
                #dataset = np.add(dataset, dataset_w2v)
                dataset_w2v = None # end of using
        elif args.vec == 4 or 4 in args.vec:
            dataset = texts2TFIDFvecs(tweetTexts_all + seedTweets, args.tfidf_dict, args.tfidf_corpus)
            seedTweetVecs = dataset[range(-20, 0), :]
            dataset = dataset[:-20, :]

        dataset = dataset.astype(np.float32)
        print "## Dataset vector obtained. ", time.asctime()

    ##############

    ##############
    # filtering tweets, clustering
    if args.do_cls:
        dayClusters = []
        #for day in dayArr:
        for day in validDays:
            tweetFCSeqIdArr = [docid for docid, dateItem in seqDayHash.items() if dateItem == day]
            texts_day = [tweetTexts_all[seqid] for seqid in tweetFCSeqIdArr]
            dataset_day = dataset[tweetFCSeqIdArr, :]

            clusterArg = getClusteringArg(args, len(tweetFCSeqIdArr))
            print "## Begin clustering in ", day, " #tweet, #vecDim", dataset_day.shape, " algorithm", args.cluster, " clusterArg", clusterArg

            tweetClusters = clusterTweets(args.cluster, texts_day, dataset_day, clusterArg)
            cLabels, tLabels, centroids, docDist = tweetClusters
            print "## Clustering done. ", " #cluster", len(cLabels), time.asctime()

            dayClusters.append((day, tweetClusters))
            #print len(dayClusters[0])
        if len(dayClusters) > 0:
            #print len(dayClusters[0])
            clusterFile = open(clusterFilePath, "w")
            cPickle.dump(dayClusters, clusterFile)
            clusterFile.close()
            print "## Clustering results stored.", clusterFilePath, time.asctime()
    else:
        clusterFile = open(clusterFilePath, "r")
        dayClusters = cPickle.load(clusterFile)
        clusterFile.close()
        print len(dayClusters[0])
        print "## Clustering results loaded.", clusterFilePath, time.asctime()

    if args.use_zs:
        dayOutClusters = []
        for dayClusterItem in dayClusters:
            print len(dayClusterItem)
            #print dayClusterItem[0]
            #print len(dayClusterItem[1]), dayClusterItem[1][0]
            #print len(dayClusterItem[2]), dayClusterItem[2][0]
            #print len(dayClusterItem[3]), dayClusterItem[3]
            #(day, texts_day, dataset_day, tweetClusters) = dayClusterItem
            (day, tweetClusters) = dayClusterItem
            if day not in validDays: continue
            dayInt = int(day)-1

            cLabels, tLabels, centroids, docDist = tweetClusters
            print "## Clustering obtained. ", clusterFilePath, " #cluster", len(cLabels), time.asctime()

            # calculate centroids nnDF/zscore in timeWindow
            #if Para_test == "4":
            if args.vec!=4 or 4 not in args.vec:
                ngIdxArray = getSim_dense(day, centroids, dataset, args.rdist, validDayWind[dayInt], dayRelWindow[dayInt])
            else:
                ngIdxArray = getSim_sparse(day, centroids, dataset, args.rdist, validDayWind[dayInt], dayRelWindow[dayInt])
            #ngIdxArray, indexedInCluster, clusters = getSim_falconn(dataset, thred_radius_dist, args.lsh_probes, None, validDayWind, dayRelWindow)
            simDfArr = getDF(day, ngIdxArray, seqDayHash, timeWindow)
            zscoreArr = getBursty(simDfArr, dayTweetNumHash, day, timeWindow)
            print "## Cluster zscore calculating done.", time.asctime()

            tweetFCSeqIdArr = [docid for docid, dateItem in seqDayHash.items() if dateItem == day]
            texts_day = [tweetTexts_all[seqid] for seqid in tweetFCSeqIdArr]
            dataset_day = dataset[tweetFCSeqIdArr, :]

            clusterFeatures = getClusterFeatures(tweetClusters, texts_day, dataset_day, seedTweetVecs, snp_comp, symCompHash)
            #docDist, cDensity, cTexts, cComps, cDocs_zip, cDistToST = clusterFeatures
            print "## Cluster zscore calculating done.", time.asctime()

            clusterScore, clusterScoreDetail = clusterScoring(tweetClusters, clusterFeatures, zscoreArr)
            print "## Clustering scoring done.", time.asctime()

            sumFlag = 3
            outputClusters = clusterSummary(sumFlag, clusterScore, cLabels, tLabels, dataset_day, args.kc, args.kt)
            print "## Clustering summary done.", time.asctime()

            if 1:
                outputTCluster(outputClusters, texts_day, clusterScoreDetail)

            dayOutClusters.append((day, texts_day, dataset_day, outputClusters))
        clusterOutputFile = open(eventFilePath+".zs", "w")
        cPickle.dump(dayOutClusters, clusterOutputFile)
        clusterOutputFile.close()
        print "## Clustering results stored.", clusterFilePath+args.zs_file, time.asctime()
    else:
        clusterOutputFile = open(eventFilePath+".zs", "r")
        dayOutClusters = cPickle.load(clusterOutputFile)
        clusterOutputFile.close()


    ##############
    ## evaluation and output
    evalOutputEvents(dayOutClusters, outputDays, devDays, testDays, args.kc, args.kc_step, Para_newsDayWindow, newsSeqDayHash, vecNews, dayNews, newsSeqComp, snp_comp, symCompHash)

    print "Program ends at ", time.asctime()
