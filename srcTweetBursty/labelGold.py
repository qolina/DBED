#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import re
import sys
import commands
import subprocess
import string
import cPickle

from nltk.tokenize import sent_tokenize

sys.path.append("../src/")
from util import snpLoader
from util import stringUtil as strUtil

#################
## preprocess: make a tmp dir
def makeTempDir(stock_tmpDir):
    if os.path.exists(stock_tmpDir):
        for tmpfile in os.listdir(stock_tmpDir):
            os.remove(stock_tmpDir + tmpfile)
    else:
        os.mkdir(stock_tmpDir)

def delHeadofContent(line):
    timeWords = ["April", "May"]
    timeWords.extend([str(x) for x in range(1, 32)])
    words = line.split(" ")
    upperLocs = [word.isupper() for word in words]
    timeIdxes = [word in timeWords for word in words]
    comb = [upperLocs[idx] ^ timeIdxes[idx] for idx in range(len(words))]
    #print upperLocs
    #print timeIdxes
    if False not in comb:
        return line
    words = words[comb.index(False):]
    if words[0] in ["", "-", "."]:
        words = words[1:]
    return " ".join(words)

## example  0.748: (Shares; fall; 8 pct)[enabler]
def load_triple(line):
    score = line[:line.index(": (")]
    triple = line[line.index(": (")+3:line.rfind(")")].split("; ")
    #if len(triple) != 3: # hard to split triple
    return float(score), triple[:3]

def load_oie_triples(oie_day_filename):
    contents = open(oie_day_filename, "r").readlines()
    contents_perLine = "".join(contents).split("\n\n")
    contents_perNews = []
    for newsIdx in range(len(contents_perLine)/2):
        contents_perNews.append(contents_perLine[newsIdx*2]+"\n"+contents_perLine[newsIdx*2+1])

    triples_perNews = []
    for content in contents_perNews:
        tripleLines = [line.lower() for line in content.split("\n") if re.match("0\.[\d]+: \(", line)]
        triples_oneNews = [load_triple(line) for line in tripleLines]
        if len(triples_oneNews) < 1: continue
        triples_perNews.append(triples_oneNews)
    return triples_perNews

def get_valid_news_content(content):
    valid_contentLines = []
    content = [line.strip() for line in content.split("\n")]
    content = [line.replace("* ", ". ") for line in content if len(line) > 1]
    if len(content) <= 5:
        return None

    valid_contentLines.append(content[0][3:])
    #valid_contentLines.append(content[3]) # filename for debug

    ##### delete special words in the first line of content
    firstLine = content[4]
    content[4] = delHeadofContent(firstLine)

    sents = sent_tokenize(" ".join(content[4:]))
    valid_contentLines.extend(sents)

    return valid_contentLines


def get_valid_1stpara_news(content):

    valid_contentLines = []
    content = [line.strip() for line in content.split("\n")]
    headline = re.sub("-", " ",content[0][3:])
    valid_contentLines.append(headline)

    content = content[4:]
    content = [line.replace("* ", ". ") for line in content]
    content.append('')
    for i in range(len(content)-1):
        if (i > 0) & (len(content[i]) > 0) & (len(content[i+1]) == 0):
            firstPara = " ".join(content[:i+1]).strip()
            firstPara = delHeadofContent(firstPara)
            valid_contentLines.extend(sent_tokenize(firstPara))
            break

    #print valid_contentLines
    return valid_contentLines

stock_newsDir = '../ni_data/stocknews/'
snpFilePath = "../data/snp500_sutd"
###############################################################
if __name__ == "__main__":
    sym_names = snpLoader.loadSnP500(snpFilePath)
    ############ prepare file to be OIE analysed
    stock_tmpDir = stock_newsDir + 'tmp/'
#    makeTempDir(stock_tmpDir)

    for dayDir in sorted(os.listdir(stock_newsDir)):
        #if dayDir < "2015-05-14": continue
        if dayDir.endswith("tmp"): continue
        #if dayDir != "2015-04-30": continue
        dayContent = set()

        for newsfile in sorted(os.listdir(stock_newsDir + dayDir)):
            #print "##############################################################"
            content = open(stock_newsDir + dayDir + "/" + newsfile, "r").read()
            printable = set(string.printable)
            content = filter(lambda x:x in printable, content)
            #print content
            #print "##############################################################"

            #sents = get_valid_news_content(content)
            sents = get_valid_1stpara_news(content)
            if sents is not None:
                headline = re.sub("^(rpt )?update\s*\d+\s", "", "####".join(sents[:2]).lower())
                dayContent.add(headline)
                #for sent in sents[:2]:
                #    sent = re.sub("^(rpt )?update\s*\d+\s", "", sent.lower())
                #    dayContent.add(sent)
                    #dayContent.append(sent)
            
        #content = [str(idx+1)+"\t"+dayContent[idx] for idx in range(len(dayContent)) if len(dayContent[idx]) < 500] # for tool clausie

        ## for debug [begin]
        #continue
        ## for debug [end]

        line_newsfilename = stock_tmpDir + dayDir
        output_file = open(line_newsfilename, "w")
        output_file.write("\n".join(dayContent).replace("####", "\n"))
        output_file.close()
        print "## File written. (temporary lined newsfile)", line_newsfilename

        ####### Apply oie tool
        oie_filename = stock_tmpDir + "oie-" + dayDir 

        ## clausie [exceptions]
        #clausie_commandline = "java -jar ../clausie/clausie.jar -vlf " + line_newsfilename + " -o " + oie_filename
        #clausie_commandline = "sh ../clausie/clausie.sh -vlf " + line_newsfilename + " -o " + oie_filename

        ## ollie
        ollie_commandline = "java -jar ../tools/ollie/ollie-app-latest.jar " + line_newsfilename + " > " + oie_filename
        os.system(ollie_commandline)
        os.remove(line_newsfilename)

        print "## File obtained. (news oie output)", oie_filename
        print "## File removed. (temporary lined newsfile)", line_newsfilename


        ## read triples
        triples_perNews = load_oie_triples(oie_filename)
        snp_syms = [snpItem[0] for snpItem in sym_names]
        snp_comp = [strUtil.getMainComp(snpItem[1]) for snpItem in sym_names]
        conf_score = 0.6
        #comp_triples = set([comp+"--"+"###".join(triple) for comp in snp_comp for triples_oneNews in triples_perNews for (score, triple) in triples_oneNews if " "+comp+" " in " ".join(["", triple[0], triple[2], ""]) and score > conf_score])
        #print "#sym_tri, comp_tri", len(sym_triples), len(comp_triples)

        comp_triples_perNews = []
        for triples_oneNews in triples_perNews:
            comp_triples = set([comp+"--"+"###".join(triple) for comp in snp_comp for (score, triple) in triples_oneNews if " "+comp+" " in " ".join(["", triple[0], triple[2], ""]) and score > conf_score])
            if len(comp_triples) < 1: continue
            comp_triples_perNews.append(comp_triples)

        ## save to file
        snp_trifile = open("../data/snp/snp_triple_in1st_"+dayDir, "w")
        #cPickle.dump(comp_triples, snp_trifile)
        cPickle.dump(comp_triples_perNews, snp_trifile)
        print "## File saved (news triples file)", snp_trifile.name, "#sym_tri, #comp_tri", 0, sum([len(item) for item in comp_triples_perNews]), "in #news", len(comp_triples_perNews)

