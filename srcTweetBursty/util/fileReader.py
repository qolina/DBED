import re
import os
import sys
import cPickle

homeDir = os.path.expanduser("~")
sys.path.append(homeDir + "/Scripts/")
import readConll

##########################################################################
#string util functions
def delNuminText(tweet_text):
    tweet_text = re.sub("\|", " ", tweet_text) # could read segged text
    words = tweet_text.split(" ")
    #words = [word for word in words if re.search("[0-9]", word) is None]
    words = [word for word in words if re.search("http", word) is None]
    tweet_text = " ".join(words)
    return tweet_text


##########################################################################
#data reading functions
def loadTweets(textFileName):
    textHash = {}
    content = open(textFileName, "r").readlines()
    content = [line[:-1].split("\t") for line in content]
    for tweet in content:
        #textHash[tweet[0]] = delNuminText(tweet[-1])
        if len(tweet[-1].split()) <3: continue
        textHash[tweet[0]] = tweet[-1]
    #print "##End of reading file.[segged text file] total lines: ", len(content), textFileName
    return textHash

def getDepLink_mwe(parsedTextFileName, textFileName):
    sentences_conll = readConll.read_conll_file(parsedTextFileName)

    dep_link_list = readConll.get_dep_links(sentences_conll)
    mwes_sents_hash = readConll.get_mwes(sentences_conll)

    content = open(textFileName, "r").readlines()
    tweet_ids = [line[:-1].split("\t")[0] for line in content if len(line) > 1]

    #print len(tweet_ids), len(mwes_sents_hash), len(dep_link_list)
    depLinkHash = dict([(tweet_ids[i], dep_link_list[i]) for i in range(len(tweet_ids))])
    mweHash = dict([(tweet_ids[i], mwes_sents_hash[i]) for i in mwes_sents_hash])

    return depLinkHash, mweHash

### not used currently
def loadnonengtext(textfilename):
    textfile = file(textfilename)
    texthash = {} # tid:text

    lineidx = 0
    while 1:
        try:
            linestr = cPickle.load(textfile)
        except eoferror:
            print "##end of reading file.[noneng text file] total lines: ", lineidx, textfilename
            break
        linestr = linestr.strip()
        lineidx += 1

        [tweet_id, tweet_text] = linestr.split("\t")
        tweet_text = delnumintext(tweet_text)

        texthash[tweet_id] = tweet_text
    textfile.close()
    return texthash
 


