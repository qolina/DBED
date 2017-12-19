# serves as sub model for OllieTwi
# ranking pivot triples for each company

import re
import sys
sys.path.append("../../Scripts/")
import hashOperation as hashOp

#import editdistance as levDis
from nltk.stem.porter import *
global stemmer
stemmer = PorterStemmer()

#@profile
def pivotRanking(snp_symname, snpTripleHash, snpTweetHash, tweetArr):
    snpTripleNumHash = dict([(snpId, len(snpTripleHash[snpId])) for snpId in snpTripleHash])
    snpSortedByTriNum = hashOp.sortHash(snpTripleNumHash, 1, True)
    stemmedWordTweetArr = [str2stemmedArr(tweet) for tweet in tweetArr]
    for snpId, triNum in snpSortedByTriNum[:1]:
        sym, compName = snp_symname[snpId]
        print "*************************************", sym, compName
        triples = snpTripleHash[snpId]
        tweetIds = snpTweetHash[snpId]
        tweets = [tweetArr[tid] for tid in tweetIds]
        stemmedTweets = [stemmedWordTweetArr[tid] for tid in tweetIds]
        stemmedWordTripleArr = [tri2stemmedArr(triple[1]) for triple in triples] # stem(" ".join(triple))
        stemmedTripleArr = [(str2stemmedArr(triple[1][0]), str2stemmedArr(triple[1][1]), str2stemmedArr(triple[1][2]))for triple in triples]

        ################
        # for debugging separate scores
        #scoreArr = [tripleBursty(triple, stemmedTweets) for triple in stemmedWordTripleArr]
        #scoreArr = [tweetsSupportTriple(triple, stemmedTweets) for triple in stemmedWordTripleArr]
        #scoreArr = [triplesSupportTriple(triple, stemmedTripleArr) for triple in stemmedTripleArr]
        #strTriples = ["###".join(triple[1]) for triple in triples]
        #scoreHash = dict(zip(strTriples, scoreArr))
        #hashOp.output_sortedHash(scoreHash, 1, True)
        #print hashOp.statisticHash(scoreHash, [0, 1, 2, 3, 4, 5])
        ################

        #pivotBurstinessScoreArr = [tripleBursty(triple, stemmedTweets) for triple in stemmedWordTripleArr]
        #tweetSupportScoreArr = [tweetsSupportTriple(triple, stemmedTweets) for triple in stemmedWordTripleArr]
        #pivotCoherenceScoreArr = [triplesSupportTriple(triple, stemmedTripleArr) for triple in stemmedTripleArr]

def tweetsSupportTriple(triple, tweets):
    # in TverskySim, tweet served as prototype, small alpha
    alpha = 0.2
    simScores = [TverskySim(triple, tweet, alpha) for tweet in tweets]
    #simScores = [max(len(tri2Str(triple)), len(tweet)) - LevenshteinDis(tri2Str(triple), tweet) for tweet in tweets]
    return sum(simScores)/len(simScores)


# triple: (score, (s, v, o))
def triplesSupportTriple(triple, triples):
    simScores = [tripleSim(triple, triItem) for triItem in triples]
    return sum(simScores)

def tri2Str(triple):
    stringTriple = " ".join(triple).strip("$#")
    #stringTriple = re.sub(" \$ ", " $", stringTriple)
    #stringTriple = re.sub(" \# ", " #", stringTriple)
    stringTriple = re.sub("\$ |\# |'s ", "", stringTriple)
    return stringTriple

def str2stemmedArr(tweet):
    return [stemmer.stem(item.lstrip("$#")) for item in tweet.split()]

def tri2stemmedArr(triple):
    return [stemmer.stem(item.lstrip("$#")) for item in tri2Str(triple).split() if item != "be"]


# tri: (stemmedS, stemmedV, stemmedO)
def tripleSim(tri1, tri2):
    #combinedFeature: (S1, S2), (V1, V2), (O1, O2), (S1V1, S2V2), (V1O1, V2O2), (S1V1O1, S2V2O2)
    combinedTriPair = [(tri1[0], tri2[0]),
            (tri1[1], tri2[1]),
            (tri1[2], tri2[2]),
            (tri1[0] + tri1[1], tri2[0] + tri2[1]),
            (tri1[1] + tri1[2], tri2[1] + tri2[2]),
            (tri1[0] + tri1[1] + tri1[2], tri2[0] + tri2[1] + tri2[2])]
    combinedTriSims = [JaccardSim(triPair[0], triPair[1]) for triPair in combinedTriPair]
    return sum(combinedTriSims)/len(combinedTriSims)


# approximate counting of frequency of Ollie triple, because it may generate new words to triple. E.G., add word "be", split "$, #" from cashtag and hashtag, or delete "$, #" from them
#@profile
def tripleBursty(triple, tweets):
    #containCount = [1 for tweet in tweets if tri2Str(triple) in tweet] # large fraction on unmatched
    #containCount = [1 for tweet in tweets if TverskySim(tri2Str(triple).split(), tweet.split(), 1)>0.7]
    containCount = [1 for tweet in tweets if TverskySim(triple, tweet, 1) > 0.9]
    return sum(containCount)


# Jaccard Coefficient
# jc(s1, s2) = |s1 & s2| / |s1|s2|
def JaccardSim(firstArr, secondArr):
    joinList = set(firstArr) & set(secondArr)
    unionList = set(firstArr) | set(secondArr)
    return len(joinList)*1.0/len(unionList)

# Tversky index
# |X&Y| / (|X&Y| + a*|X-Y| + b*|Y-X|)
# suitable assign:  a + b = 1
def TverskySim(firstArr, secondArr, alpha):
    firstArr = set(firstArr)
    secondArr = set(secondArr)
    joinList = firstArr & secondArr
    fstNoSndList = firstArr.difference(secondArr)
    sndNoFstList = secondArr.difference(firstArr)
    #print len(firstArr), len(secondArr), len(joinList), len(fstNoSndList), len(sndNoFstList)
    return len(joinList)*1.0/ (len(joinList) + alpha*len(fstNoSndList) + (1-alpha)*len(sndNoFstList))

# Levenshtein Distance  : a slow version
def LevenshteinDis(str1, str2):
    i = len(str1)
    j = len(str2)

    if min(i, j) == 0:
        return max(i, j)
    else:
        dis1 = LevenshteinDis(str1[:-1], str2)
        dis2 = LevenshteinDis(str1, str2[:-1])
        dis3 = LevenshteinDis(str1[:-1], str2[:-1])
        if str1[i-1] != str2[j-1]:
            return min(dis1+1, dis2+1, dis3+1)
        else:
            return min(dis1+1, dis2+1, dis3)

##########################################
### main
if __name__ == "__main__":
    # testing for levenshteindis - a fast python package
    #print levDis.eval("banana", "bahama")

    triple = ("botched apple 's second flop", "is at", "hand photo epa $")
    tweet = "Botched Apple's second flop is at hand Photo EPA $aapl".lower()
    triple = ("recent leaders", "showed up in", "yday 's most bearish list thinning market $ ibb $ aapl $ bidu $ fb")
    tweet = "Recent leaders showed up in yday's most bearish list Thinning market $IBB $AAPL $BIDU $FB".lower()
    print tri2stemmedArr(triple)
    print str2stemmedArr(tweet)
    print TverskySim(tri2stemmedArr(triple), str2stemmedArr(tweet), 1)
