import sys
import time
from collections import Counter

import numpy as np
from sklearn.metrics import pairwise
from zpar import ZPar
from nltk.stem.porter import PorterStemmer

from labelGold import *
from tweetVec import getVec
from pivotRanking import JaccardSim

from tweetClustering import findComp_name, compInCluster, topCompInCluster

def debugMainNameComp(snp_comp):
    snp_comp1 = [sym_names[i][0]+"\t"+sym_names[i][1]+"\t"+snp_comp[i] for i in range(len(snp_comp)) if len(snp_comp[i].split()) == 1]
    snp_comp2 = [sym_names[i][0]+"\t"+sym_names[i][1]+"\t"+snp_comp[i] for i in range(len(snp_comp)) if len(snp_comp[i].split()) == 2]
    snp_comp3 = [sym_names[i][0]+"\t"+sym_names[i][1]+"\t"+snp_comp[i] for i in range(len(snp_comp)) if len(snp_comp[i].split()) > 2]
    print len(snp_comp1), len(snp_comp2), len(snp_comp3)
    print "\n".join(sorted(snp_comp1))
    print "\n".join(sorted(snp_comp2))
    print "\n".join(sorted(snp_comp3))

def frmTriple(parsed_sents, comps):
    #matchedComps = [item[0] for item in comps]
    #print matchedComps
    frmValid = set()
    for parsed_sent in parsed_sents:
        #print "#####################"
        #print parsed_sent
        items = parsed_sent.strip().split("\n")
        items = [item.strip().split("\t") for item in items]
        words = [item[0] for item in items]
        tags = [item[1].lower() for item in items]
        links = [int(item[2]) for item in items]
        deps = [item[3].lower() for item in items]
        sent = " ".join(words)
        # ori
        #valid_words = [words[idx] for idx, tag in enumerate(tags) if tag[:2] in ["nn", "vb", "jj"] if deps[idx] in ["root", "sub", "obj", "vc", "vmod", "nmod", "pmod"]]

        # loose1
        valid_words = [words[idx] for idx, tag in enumerate(tags) if tag[:2] in ["nn", "vb", "jj", "cd", "rb"] if deps[idx] in ["root", "sub", "obj", "vc", "vmod", "nmod", "pmod"]]

        # loose2
        #valid_words = [words[idx] for idx, tag in enumerate(tags) if tag[:2] in ["nn", "vb", "jj", "rb"] if deps[idx] in ["root", "sub", "obj", "vc", "vmod", "nmod", "pmod"]]
        frmValid.update(set(valid_words))
        #print valid_words
        continue

        # use dep link
        #for comp in matchedComps:
        #    compLen = len(comp.split())
        #    if compLen == 1:
        #        idxs = [words.index(comp)]
        #    else:
        #        idx = sent.find(comp)
        #        if idx < 0: continue
        #        preWordNum = len(sent[:idx].strip().split())
        #        idxs = range(preWordNum, preWordNum+compLen)# words[idxs[0]:idxs[-1]+1]
        #    print comp, compLen, idxs, words[idxs[0]:idxs[-1]+1]
        #    linksTo = [links[idx] for idx in idxs]
        #    linksFrom = [idx for idx, link in enumerate(links) if link in idxs or link==-1]
        #    linksFrom.extend(linksTo)

        #    linksTo_2 = [links[idx] for idx in linksFrom if links[idx]!=-1]
        #    linksFrom_2 = [idx for idx, link in enumerate(links) if link in linksFrom]
        #    linksFrom.extend(linksTo_2)
        #    linksFrom.extend(linksFrom_2)
        #    #valid_links = [(idx, words[idx], deps[idx]) for idx in linksFrom if deps[idx].lower() in ['sub', 'obj', 'root']]
        #    valid_links = [(idx, words[idx], deps[idx]) for idx in linksFrom]
        #    print valid_links
        #    valid_links = [(idx, words[idx], deps[idx]) for idx in linksFrom if tags[idx].startswith("nn") or tags[idx].startswith("vb")]
        #    print valid_links
    return list(frmValid)


#[day_0430, day_0501, ...]
def extractStockNews(stock_newsDir, symCompHash, sentNUM):
    snp_comp = symCompHash.values()
    zparModel = ZPar('english-models')
    #tagger = zparModel.get_tagger()
    depparser = zparModel.get_depparser()
    stemmer = PorterStemmer()

    dayNews = []
    for dayDir in sorted(os.listdir(stock_newsDir)):
        if len(dayDir) != 10: continue
        #if int(dayDir[-2:]) > 5: continue
        #if dayDir != "2015-04-30": continue
        newsContents = set()
        for newsfile in sorted(os.listdir(stock_newsDir + dayDir)):
            #print "##############################################################"
            content = open(stock_newsDir + dayDir + "/" + newsfile, "r").read()
            printable = set(string.printable)
            content = filter(lambda x:x in printable, content)
            #print content
            #print "##############################################################"

            #sents = get_valid_news_content(content)
            sents = get_valid_1stpara_news(content)
            if sents is None: continue
            headline = re.sub("^(rpt )?update\s*\d+\s", "", "###".join(sents[:sentNUM]).lower())
            headline = re.sub("\s+", " ", headline)
            newsContents.add(headline)

        oneDayNews = [] # [(matchedSNPComp, headline), ...]
        # matchedSNPComp: [(matchedPart, WholeCompName), ...]
        fullNameNum = 0
        doubtCounter = Counter()

        if 0:
            print "\n".join(sorted(list(newsContents)))
            continue

        newsHash = {}
        headlineCompHash = {}

        for headline in newsContents:
            fullMatch = findComp_name(headline.replace("###", " "), snp_comp)
            #symMatch = [(word, symCompHash[word]) for word in headline.replace("###", " ").split() if word in symCompHash and word not in ["a", "an", "has"]]
            symMatch = [word for word in headline.replace("###", " ").split() if word in ["ge", "gt", "gm", "aig", "cvs", "oi", "adm", "jpm", "twc", "cvc", "se"]]
            symMatch = list([symCompHash[sym] for sym in set(symMatch)])
            if fullMatch is not None or len(symMatch) > 0:
                if 0:
                    print "---------------------------"
                    print fullMatch, symMatch
                    print headline
                    continue

                if fullMatch is not None:
                    symMatch.extend(fullMatch)

                headlineCompHash[headline] = symMatch

                # get valid words in headline
                parsed_sents = [depparser.dep_parse_sentence(sent) for sent in headline.split("###")]
                triples = frmTriple(parsed_sents, None)
                triples = [stemmer.stem(word) for word in triples if word not in [":", "(", ")", ",", ".", "\"", "'"]]
                sortedText = " ".join(sorted(triples))
                if sortedText not in newsHash:
                    newsHash[sortedText] = headline

        for impText, headline in newsHash.items():
            fullNameNum += 1
            oneDayNews.append((headlineCompHash[headline], headline, impText.split()))


            #doubtMatch = [matchedComp[idx] for idx in range(len(matchedComp)) if matchScore[idx] > 0.33 and matchScore[idx] < 0.66]
            #wrongMatch = [matchedComp[idx] for idx in range(len(matchedComp)) if matchScore[idx] <= 0.33]

        #print "full", fullNameNum, len(newsContents), round(float(fullNameNum)/len(newsContents), 2)
        print "## Stock news extracting done in day", dayDir, " #snp_matched", fullNameNum, " out of all", len(newsContents), time.asctime()
        dayNews.append(oneDayNews)
        #break
    return dayNews

def content2vec(word2vecModelPath, dayNews):
    contentNews = []
    dayNewsNum = []
    newsSeqDayHash = {} # newsSeqId in contentNews: day_int
    newsSeqComp = {} # newsSeqId in contentNews: matchComp
    for dayIdx, oneDayNews in enumerate(dayNews):
        newsHeadlines = [item[1].replace("###", " ") for item in oneDayNews]
        matchedComps = [item[0] for item in oneDayNews]
        dayNewsNum.append(len(newsHeadlines))
        #print newsHeadlines
        for seqid, comp in enumerate(matchedComps, start=len(contentNews)):
            newsSeqDayHash[seqid] = dayIdx
            newsSeqComp[seqid] = comp
        contentNews.extend(newsHeadlines)
    vecNews = getVec('3', None, None, None, word2vecModelPath, contentNews)
    return vecNews, newsSeqDayHash, newsSeqComp

def stockNewsVec(stock_newsDir, symCompHash, word2vecModelPath, newsVecPath, sentNUM):
    dayNews = extractStockNews(stock_newsDir, symCompHash, sentNUM)
    #return 0
    vecNews, newsSeqDayHash, newsSeqComp = content2vec(word2vecModelPath, dayNews)
    outputFile = open(newsVecPath, "w")
    cPickle.dump(dayNews, outputFile)
    cPickle.dump(vecNews, outputFile)
    cPickle.dump(newsSeqDayHash, outputFile)
    cPickle.dump(newsSeqComp, outputFile)

###########################################################
## nns_fromSim is groudTruth nearest neighbors
# result:
#                    20_ng,1k_data  10_ng,10k_data
# nnModel, auto:        0.3465      0.3339
#          ball_tree:   0.3465      0.3339
#          kd_tree:     0.3465      0.3339
#          lshforest:   0.3819      0.3204
def eval_sklearn_nnmodel(nns_fromSim, rngIdxArray):
    # test nearest neighbor
    evalNNModel_scoreArr = []
    for docid in range(rngIdxArray.shape[0]):
        nn_onedoc_sim = [item[0] for item in nns_fromSim[docid]]
        nn_onedoc_nnmodel = list(rngIdxArray[docid])
        score = JaccardSim(nn_onedoc_sim, nn_onedoc_nnmodel)
        evalNNModel_scoreArr.append(score)

        if docid in range(3):
            common = set(nn_onedoc_nnmodel) & set(nn_onedoc_sim)
            union = set(nn_onedoc_nnmodel) | set(nn_onedoc_sim)
            print "#### docid", docid
            print common
            print set(nn_onedoc_sim)-common
            print set(nn_onedoc_nnmodel)-common
            print union
            print len(nn_onedoc_sim), len(nn_onedoc_nnmodel), len(common), len(union), score
    jcSim = round(np.mean(np.asarray(evalNNModel_scoreArr)), 4)
    print "jaccard sim", jcSim
    return jcSim

def tClusterMatchNews_vec(newsSeqCompDay, vecNewsDay, vecsIn, compsIn):
    newsMatchComp = [newsIdx for comp, count in compsIn for newsIdx, newsComps in enumerate(newsSeqCompDay) for newsCompItem in newsComps if comp == newsCompItem[0]]
    simMatrix = pairwise.cosine_similarity(vecsIn, vecNewsDay)
    simToNews = np.sum(simMatrix, axis=0)
    simToNews_byComp = [(idx, simToNews[idx]) for idx in newsMatchComp if simToNews[idx] >= np.mean(simToNews)]
    #print "** ", simToNews.shape, np.max(simToNews), np.min(simToNews), np.mean(simToNews)
    if len(simToNews_byComp) > 0:
        return [idx for idx,sim in simToNews_byComp]

def normTweetWords(word):
    normFinWords = {"est":"estimate", "rev":"revenue"}
    if word in normFinWords:
        return normFinWords[word]
    return word

def tClusterMatchNews_content(newsSeqCompDay, textNewsDay, textsIn, compsIn):
    stemmer = PorterStemmer()
    matchedNews_c = []
    newsMatchComp = [newsIdx for comp, count in compsIn for newsIdx, newsComps in enumerate(newsSeqCompDay) for newsCompItem in newsComps if comp == newsCompItem[0]]
    for newsIdx in newsMatchComp:
        ncomps, nText, nValidWords = textNewsDay[newsIdx]
        nCompsWords = " ".join([comp[0] for comp in ncomps]).split()
        #print "--news", ncomps, nText, nValidWords, nCompsWords

        tWords = []
        tNumIn = sum([item[1] for item in textsIn])
        for textsItem in textsIn[:min(2, len(textsIn))]:
            text, dnum = textsItem
            if dnum < 2 or dnum < tNumIn/10: continue
            tWords.extend(text.lower().split())
        tWords = [normTweetWords(word) for word in tWords]
        tWords = set([stemmer.stem(word) for word in tWords if word not in nCompsWords])
        #print "--tweetWords stem", sorted(list(tWords))

        commonWords = (tWords & set(nValidWords))# - set(nCompsWords)
        if len(commonWords) == 0: continue

        #print newsIdx, commonWords
        matchedNews_c.append(newsIdx)

    if len(matchedNews_c) == 0:
        return None
    return matchedNews_c

def tClusterMatchTripNews(textNewsDay, textsIn, compsIn):
    matchedTrip = []
    matchedNews_c = []
    for docIdx, textsItem in enumerate(textsIn):
        text, dnum = textsItem
        #if dnum < 2: continue
        compsInText = [comp for comp, cnum in compsIn if text.lower().find(comp) >= 0 and cnum>=dnum]
        if len(compsInText) < 1: continue

        #print "compsInText", compsInText
        for newsIdx, tripNews in enumerate(textNewsDay):
            tripsHasComp = [(comp, trip.split("--")[1]) for comp in compsInText for trip in tripNews if trip.split("--")[0] == comp]
            if len(tripsHasComp) < 1: continue
            #print "tripsHasComp", tripsHasComp
            common = [set(trip.replace("###", " ").split())& set(text.lower().split())-set(comp.split()) for comp, trip in tripsHasComp]
            #print "common", common
            if len(common) > 2:
                matchedNews_c.append(newsIdx)

    if len(matchedNews_c) == 0:
        return None
    return matchedNews_c

def evalTClusters(tweetClusters, feaVecs, documents, vecNewsDay, textNewsDay, newsSeqCompDay, snp_comp, symCompHash, outputDetail):
    trueCluster = []
    matchedNews = []
    for outIdx, cluster in enumerate(tweetClusters):
        clabel, cscore, docsIn = cluster
        tNumIn = sum([dscore for did, dscore in docsIn])
        textsIn = [(documents[docid], num) for docid, num in docsIn]
        compsIn = compInCluster(textsIn, snp_comp, symCompHash, True, False)
        compsIn = topCompInCluster(compsIn, tNumIn, 2)

        if feaVecs is not None:
            #vecsIn = feaVecs[docsIn[0][0],:]
            #matchedNews_c = tClusterMatchNews_vec(newsSeqCompDay, vecNewsDay, vecsIn, compsIn)
            matchedNews_c = tClusterMatchNews_content(newsSeqCompDay, textNewsDay, textsIn, compsIn)
        else:
            matchedNews_c = tClusterMatchTripNews(textNewsDay, textsIn, compsIn)

        if outputDetail:
            print "############################"
            print "1-** cluster", outIdx, clabel, cscore, ", #tweet", tNumIn, compsIn
            for docid, dscore in docsIn:
                #if docsIn[0][1] > 1 and dscore < 2: continue
                print docid, dscore, "\t", documents[docid]

        if matchedNews_c is None: continue

        trueCluster.append((outIdx, matchedNews_c))
        matchedNews.extend(matchedNews_c)

        if outputDetail:
            #sortedNews = sorted(simToNews_byComp, key = lambda a:a[1], reverse = True)
            sortedNews = Counter(matchedNews_c).most_common()
            for idx, sim in sortedNews:
                print '-MNews', idx, sim, textNewsDay[idx]

    matchedNews = Counter(matchedNews)
    if 1:
        print "TrueClusters", trueCluster
        matchNewsDetails = {}
        for cid, nids in trueCluster:
            for nid in set(nids): 
                if nid in matchNewsDetails:
                    matchNewsDetails[nid].append(cid)
                else:
                    matchNewsDetails[nid] = [cid]
        print "MCNewsDetail", sorted(matchNewsDetails.items(), key = lambda a:a[0])
    return len(trueCluster), len(matchedNews)


def outputEval(Nums):
    print "## Eval newsMatchedCluster", sum(Nums[0]), sum(Nums[1]), round(float(sum(Nums[0])*100)/sum(Nums[1]), 2)
    print "## Eval sysMatchedNews", sum(Nums[2]), sum(Nums[3]), round(float(sum(Nums[2])*100)/sum(Nums[3]), 2)

def outputEval_day(Nums):
    print "## newsPre", Nums[0][-1], Nums[1][-1], ("%.2f" %(Nums[0][-1]*100.0/Nums[1][-1])), "\t",
    print "## newsRecall", Nums[2][-1], Nums[3][-1], "\t", round(float(Nums[2][-1]*100)/Nums[3][-1], 2)

def dayNewsExtr(newsDayWindow, newsSeqDayHash, vecNews, dayNews, newsSeqComp):
    newsSeqIdDay = sorted([newsSeqId for newsSeqId, dayInt in newsSeqDayHash.items() if dayInt in newsDayWindow])
    vecNewsDay = vecNews[newsSeqIdDay,:]
    textNewsDay = []
    for item in newsDayWindow:
        textNewsDay.extend(dayNews[item])
    newsSeqCompDay = [newsSeqComp[newsSeqId] for newsSeqId in newsSeqIdDay]
    return vecNewsDay, textNewsDay, newsSeqCompDay

def dayNewsTripExtr(newsDayWindow):
    compTrip_News = []
    for dayInt in newsDayWindow:
        if dayInt == 0:
            suffix = "04-30"
        else:
            suffix = "05-"+str(dayInt).zfill(2)
        tripFile = open("../data/snp/snp_triple_in1st_2015-"+suffix, "r")
        compTrip_News_day = cPickle.load(tripFile)
        compTrip_News.extend(compTrip_News_day)

    return compTrip_News

def evalOutputEvents(dayClusters, outputDays, devDays, testDays, topK_c, Kc_step, Para_newsDayWindow, newsSeqDayHash, vecNews, dayNews, newsSeqComp, snp_comp, symCompHash):
    outputDetail = False

    for sub_topK_c in range(Kc_step, topK_c+1, Kc_step):
        if sub_topK_c == topK_c: outputDetail = True

        dev_Nums = [[], [], [], []] # trueCNums, cNums, matchNNums, nNums 
        test_Nums = [[], [], [], []]
        for cItem in dayClusters:
            if cItem is None: continue
            day, texts_day, dataset_day, tweetClusters = cItem
            if tweetClusters is None: continue
            if day not in outputDays: continue

            sub_tweetClusters = tweetClusters[:sub_topK_c]

            newsDayWindow = [int(day)+num for num in Para_newsDayWindow]
            vecNewsDay, textNewsDay, newsSeqCompDay = dayNewsExtr(newsDayWindow, newsSeqDayHash, vecNews, dayNews, newsSeqComp)

            if outputDetail:
                print "## News in day", day
                for item in textNewsDay:
                    print item
                print "## Output details of Clusters in day", day

            trueCNum, matchNNum = evalTClusters(sub_tweetClusters, dataset_day, texts_day, vecNewsDay, textNewsDay, newsSeqCompDay, snp_comp, symCompHash, outputDetail)

            if day in devDays:
                dev_Nums[0].append(trueCNum)
                dev_Nums[1].append(len(sub_tweetClusters))
                dev_Nums[2].append(matchNNum)
                dev_Nums[3].append(len(textNewsDay))
                if trueCNum > 0:
                    outputEval_day(dev_Nums)
            if day in testDays:
                test_Nums[0].append(trueCNum)
                test_Nums[1].append(len(sub_tweetClusters))
                test_Nums[2].append(matchNNum)
                test_Nums[3].append(len(textNewsDay))
                if trueCNum > 0:
                    outputEval_day(test_Nums)
                
        ##############

        ##############
        # output evaluation metrics_recall
        if sum(dev_Nums[1]) > 0:
            print "** Dev exp in topK_c", sub_topK_c
            outputEval(dev_Nums)
        if sum(test_Nums[1]) > 0:
            print "** Test exp in topK_c", sub_topK_c
            outputEval(test_Nums)
        ##############


stock_newsDir = '../ni_data/stocknews/'
snpFilePath = "../data/snp500_sutd"
word2vecModelPath = "../ni_data/tweetVec/w2v1010100-en"
newsVecPath = "../ni_data/tweetVec/stockNewsVec1_Loose1"

###############################################################
if __name__ == "__main__":
    sym_names = snpLoader.loadSnP500(snpFilePath)
    snp_syms = [snpItem[0] for snpItem in sym_names]
    snp_comp = [strUtil.getMainComp(snpItem[1]) for snpItem in sym_names]
    symCompHash = dict(zip(snp_syms, snp_comp))
    #debugMainNameComp(snp_comp)
    #sys.exit(0)

    # prepare news vec for eval recall
    stockNewsVec(stock_newsDir, symCompHash, word2vecModelPath, newsVecPath, 1)
    sys.exit(0)

    # load news vec
    newsVecFile = open(newsVecPath, "r")
    dayNews = cPickle.load(newsVecFile)
    vecNews = cPickle.load(newsVecFile)
    newsSeqDayHash = cPickle.load(newsVecFile)
    newsSeqComp = cPickle.load(newsVecFile)

    day = '05'
    newsSeqIdDay = sorted([newsSeqId for newsSeqId, dayInt in newsSeqDayHash.items() if dayInt == int(day)])
    print len(newsSeqIdDay)
    print vecNews.shape
    vecNewsDay = vecNews[newsSeqIdDay,:]
    newsSeqCompDay = [newsSeqComp[newsSeqId] for newsSeqId in newsSeqIdDay]
