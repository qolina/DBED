# read formatted sentence  file (may have multiple root) 
# input: conll formatted file;
# output: 
import sys
import re
import numpy as np

def read_conll_file(filename):
    # (sent_conll) format: multi line of word_conll
    #(word_conll) format: idx, word, _, pos, pos, _, head, _, _, _
    # (ele_word) def: each item in word_conll

    # empty last line in parse file
    contents = open(filename, "r").readlines()[:-1]
    contents = [line[:-1].lower() for line in contents]

    contents = "###".join(contents).strip("###")
    contents = contents.split(r"######")
    #print len(contents)
    #print "\n".join(contents)
    sentences_conll = []
    for line in contents:
        wordItems = line.split(r"###") # each word_conllOut as one item
        wordItems = [word.split("\t") for word in wordItems]
#        print wordItems
        sentences_conll.append(np.array(wordItems))

    print "##End of reading file.[parsed text file for depLink_mwe]  total sents: ", len(sentences_conll), filename
    return sentences_conll

def get_dep_links_in_a_sent(sent):

    idxs = np.array(sent[:,0], dtype=np.int)
    dep_idxs = np.array(sent[:,6], dtype=np.int)
    np.subtract(idxs, 1, idxs)
    np.subtract(dep_idxs, 1, dep_idxs)

    dep_links = zip(idxs, dep_idxs)
    dep_links = [item for item in dep_links if item[1] >= 0]

    #print " ".join([wordItem[1] for wordItem in sent])
#    print sent
#    print sent.shape
#    print idxs
#    print dep_idxs
#    print dep_links

    return dep_links

def mwe_map(mwe_tag, idx):
    mwe_tag = list(mwe_tag)
    mwe_tag.insert(0, "_")
    if mwe_tag[idx+1] == "mwe" or mwe_tag[idx] == "mwe": # current word or preword is mwe
        return str(idx)
    return '-'

def get_mwe_in_a_sent(sent):
#    print sent
    mwe_tag = sent[:, -1]
    in_mwe = [mwe_map(mwe_tag, i) for i in range(len(mwe_tag))]
    in_mwe_str = "#".join(in_mwe)
    mwes = re.sub("(-#)+", "-#", in_mwe_str).strip("-#")
    if len(mwes) == 0:
        return []
    mwes = mwes.split("-#")
    mwes = [np.array(item.strip("#").split("#"), dtype=np.int) for item in mwes]
    return mwes

# return deplinks in sent:(word_from, word_to)
def get_dep_links(sentences_conll):
    dep_link_list = []

    for sent in sentences_conll:
        dep_links_id = get_dep_links_in_a_sent(sent)
        words = np.array(sent[:, 1])
        dep_links_word = [(words[item[0]], words[item[1]]) for item in dep_links_id] 
        dep_link_list.append(dep_links_word)

    return dep_link_list

# return mwes in sent: [mwe_word1, mwe_word2, ...]
def get_mwes(sentences_conll):
    mwe_hash = {}
    for sent_id in range(len(sentences_conll)):
        sent = sentences_conll[sent_id]
        mwes_id = get_mwe_in_a_sent(sent)
        words = np.array(sent[:, 1])
        mwes_word = []
        for mwe in mwes_id:
            mwes_word.append([words[idx] for idx in mwe])
        if len(mwes_word) > 0:
            mwe_hash[sent_id] = mwes_word
    return mwe_hash

##############
def getArg(args, flag):
    arg = None
    if flag in args:
        arg = args[args.index(flag)+1]
    return arg

def parseArgs(args):
    arg1 = getArg(args, "-in")
    if arg1 is None:
        print "Usage: python readconll.py -in conllfilename"
        sys.exit(0)
        
    return arg1

###################################
if __name__ == "__main__":

    conll_filename = parseArgs(sys.argv)

    sentences_conll = read_conll_file(conll_filename)
    dep_link_list = get_dep_links(sentences_conll)
    mwe_hash = get_mwes(sentences_conll)
    print len(dep_link_list), len(mwe_hash)


