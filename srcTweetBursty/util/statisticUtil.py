

##########################################################################
#statistic util functions
def getCooccurApp_tri(key1, key2, key3, unitAppHash):
    commonApp_1 = getCooccurApp(key1, key2, unitAppHash)
    commonApp_2 = getCooccurApp(key1, key3, unitAppHash)
    if commonApp_1 is None or commonApp_2 is None:
        return None
    commonApp = [tid for tid in commonApp_1 if tid in commonApp_2]
    return commonApp

def getCooccurCount_tri(key1, key2, key3, unitAppHash):
    commonApp = getCooccurApp_tri(key1, key2, key3, unitAppHash)
    if commonApp is None:
        return 0
    return len(commonApp)

def getCooccurCount(key1, key2, unitAppHash):
    commonApp = getCooccurApp(key1, key2, unitAppHash)
    if commonApp is None:
        return 0
    return len(commonApp)

def getCooccurApp(key1, key2, unitAppHash):
    if key1 == key2:
        return None 
    appHash1 = unitAppHash.get(key1)
    appHash2 = unitAppHash.get(key2)

    commonApp = [tid for tid in appHash1 if tid in appHash2]
    return commonApp

def g_test(sym, item, unitAppHash, Tweets_day):
    appHash_s = unitAppHash.get(sym)
    appHash_u = unitAppHash.get(item)
    appHash_su = {}
    appHash_su.update(appHash_s)
    appHash_su.update(appHash_u)

    Osu = getCooccurCount(sym, item, unitAppHash)
    Os_u = len(appHash_s) - Osu
    O_su = len(appHash_u) - Osu
    O_s_u = Tweets_day - len(appHash_su)

    oArr = [Osu, Os_u, O_su, O_s_u]
    oArr = [val*1.0/Tweets_day for val in oArr]
    e = 1.0/len(oArr)

#    print oArr
    tempScore = [val*math.log(val/e) for val in oArr]
    g_test_score = sum(tempScore)
    return g_test_score



