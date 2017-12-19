import sys
import os

from util.rank_metrics import *

def match2Lbl(line):
    if line.startswith("1-**"):
        return 1
    elif line.startswith("-**"):
        return 0

def getAnnoLbl(filename):
    annotatedFile = open(filename, 'r')
    content = annotatedFile.readlines()
    print "######## pre recall by news"
    print "".join(content[-3:-1])
    recall_news = float(content[-2].split()[-1].strip()) 

    clusterLabels = [(content[lineIdx+1], match2Lbl(line)) for lineIdx, line in enumerate(content) if line[:4] in ["1-**", "-** "]]
    return clusterLabels, recall_news

# topK_para, get topK_para events in one day
# Kc, num of total events in each day
def getLabelByDay(annotatedLabels, topK_para, dayIndex, Kc):
    labels_day = annotatedLabels[Kc*dayIndex:Kc*(dayIndex+1)][:topK_para]
    return labels_day

def evalP(annotatedLabels, preK, days, topK_c, recall_news):
    print "Eval: MAP, Prec@k, NDCG"
    for para in preK:
        metrics_para = []

        labels_para = [getLabelByDay(annotatedLabels, para, dayIndex, topK_c) for dayIndex in range(days)]

        map_para = mean_average_precision(labels_para)
        pre_para = [precision_at_k(labels_para_day, para) for labels_para_day in labels_para]
        ndcg_para = [ndcg_at_k(labels_para_day, para) for labels_para_day in labels_para]

        metrics_para.append(map_para*100)
        metrics_para.append(np.mean(pre_para)*100)
        metrics_para.append(np.mean(ndcg_para)*100)
        metrics_para = [("%.2f" %item) for item in metrics_para]

        #print "## Eval in top", para
        #print "map, pre@k, ndcg", metrics_para
        print para, "\t", metrics_para[0], "\t", metrics_para[1], "\t", metrics_para[2]

def caltopP(annotatedLabels, preK, days, topK_c, recall_news):
    pre_byDay = []

    precision = []
    for para in preK:
        trueLbl_day = [sum(annotatedLabels[topK_c*i:topK_c*(i+1)][:para]) for i in range(days)]
        pre_byDay.append([("%.2f" %(item*100.0/para)) for item in trueLbl_day])
        precision.append(round(float(sum(trueLbl_day))*100/(days*para), 2))
    print "Pre_byDay @topK", preK
    for item in pre_byDay:
        print item

    print "pre@top", preK
    print "pre", precision
    p = precision[-1]
    print "f1", 2.0*p*recall_news/(p+recall_news)

if __name__ == "__main__":
    print "Usage: python precision.py annotatedFilename topK_c"
    print sys.argv[1:3]

    filename = sys.argv[1]
    topK_c = int(sys.argv[2])
    Kc_step = 5
    clusterLabels, recall_news = getAnnoLbl(filename)
    annotatedLabels = [label for firstTweet, label in clusterLabels]

    preK = range(Kc_step, topK_c+1, Kc_step)
    days = len(annotatedLabels)/topK_c

    #caltopP(annotatedLabels, preK, days, topK_c, recall_news)
    evalP(annotatedLabels, preK, days, topK_c, recall_news)
