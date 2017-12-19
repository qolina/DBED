import sys

from precision import getAnnoLbl

if __name__ == "__main__":
    print "Usage: python annoPartial.py 2bLblFilename, lblFilename1, lblFilename2 ..."
    tblblFilename = sys.argv[1]
    lblFilenames = sys.argv[2:]
    print sys.argv[1:]

    fstTweLabels = []
    for lblFilename in lblFilenames:
        clusterLabels, recall_news = getAnnoLbl(lblFilename)
        fstTweLabels.extend([(" ".join(firstTweet.split(" ")[2:]).lower(), lbl) for firstTweet, lbl in clusterLabels])
    fstTweLabels = dict(fstTweLabels)

    tbContent = open(tblblFilename, "r").readlines()

    processedTBContent = []
    for lineIdx, line in enumerate(tbContent):
        if line.startswith("1-**"):
            st = line.find("#tweet")+6
            tweetNum = int(line[st:line.find("[", st)].strip())
            firstTweet = tbContent[lineIdx+1]
            #tIdx = firstTweet[:firstTweet.find(" ")]
            #print firstTweet
            tNum = int(firstTweet.split(" ")[1])
            if tNum < tweetNum/3: lbl = None
            else:
                tText = " ".join(firstTweet.split(" ")[2:]).lower()
                lbl = fstTweLabels.get(tText)
            if lbl is None:
                lbl = "-1"
            elif lbl == 0:
                lbl = ""
            else:
                lbl = str(lbl)

            processedTBContent.append(lbl+line[1:])
        else:
            processedTBContent.append(line)

    outFile = open(tblblFilename+".al", "w")
    outFile.write("".join(processedTBContent))
    outFile.close()
