import re
import sys
import os


sys.path.append(os.path.expanduser("~") + "/Scripts/")
import hashOperation as hashOp

##########################################################################
#string util functions

### extract mainName out of company fullname
def getMainComp(compName):
    mainName = re.sub(r"\b(?:corporation|incorporated|technologies|limited|company|group|inc|corp|ltd|svc.gp.|int'l|the|plc|cos|co)\b[.]?", "", compName)
    #mainName = re.sub(r"\b(?:laboratories|international|resources|systems|sciences|devices)\b[.]?", "", mainName)
    mainName = re.sub(r"\b(?:laboratories|energy|financial|analytics|communities|brands|companies|resources|system|systems|national|sciences|devices|banks|bank|holdings|technology)$", "", mainName.strip())
    mainName = re.sub(r"\b(class .)|(\.com)\b", "", mainName)
    mainName = re.sub("\s+", " ", mainName).strip()
    repair = [("united", "technologies"), ("news", "corporation"), ("target", "corporation"), ("regions", "financial"), ("ball", "corporation"), ("southern", "company"), ("range", "resources"), ("l", "brands"), ("linear", "technology")]
    repaired = [mainName+" "+item[1] for item in repair if item[0]==mainName]
    if len(repaired) >= 1:
        mainName = repaired[0]
    return mainName.strip(",&! ")

def comps_in_a_sent(sentence):
    comps = [1 for word in sentence.split() if word.startswith("$")]
    return comps

def sym_start(sentence, sym, compName):
    if sentence.startswith(sym):
        return 1

    firstWord_comp = compName.split(" ")[0]
    if sentence.startswith(firstWord_comp):
        return 1
    return 0

# sym: target entity (snp company: $+abbrv_name)
# word: bursty verb 
# depLinks: dependency link appeared in one sentence
def hasDepLink(sym, word, depLinks):
    sym = sym[1:] # del first letter($)
    comb_of_sym_word = [(sym, word), (word, sym)]

    company_fullname = nameHash.get(sym).split(" ")
    comb_of_full_ele_word = [(item, word) for item in company_fullname]
    comb_of_full_ele_word_inv = [(word, item) for item in company_fullname]


def contain_depLink_sent(compWords, word, depLinks, mwes):
    #### consider mwes
    if mwes is not None:
        result = [compWords.extend(mwe) for mwe in mwes]
    compWords = set(compWords)

    wordpair = [(cmpword, word) for cmpword in compWords]
    wordpair_inv = [(word, cmpword) for cmpword in compWords]
    wordpair.extend(wordpair_inv)
#    print wordpair

    if hashOp.hasSameKey_le1_inlist(depLinks, wordpair) > 0:
        return True
    return False


