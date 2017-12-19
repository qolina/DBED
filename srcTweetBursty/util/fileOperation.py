import time
import os
import sys

## load stop words
def loadStopword(stopwordsFilePath):
    stopwordHash = {}
    stopFile = file(stopwordsFilePath)
    while True:
        lineStr = stopFile.readline()
        if not lineStr:
            print "### " + str(time.asctime()) + " #" + str(len(stopwordHash)) + " stop words are loaded from " + stopwordsFilePath
            break
        stopwordHash[lineStr[:-1].strip()] = 1
    stopFile.close()
    return stopwordHash


