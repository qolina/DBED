# this file contain multiple functions on python dictionary

def cumulativeInsert(hash, item, val):
    if item in hash:
        hash[item] += val
    else:
        hash[item] = val
    return hash


def count2Ratio(hash):
    N = sum(hash.values()) 
    hash = dict([(item, float("%.3f"%(hash[item]*100.0/N))) for item in hash])
    return N, hash


# create a 3-member tuple, key, value_in_sortedList, value_in_hash
def joint_with_sortedHash(sortedList, hash):
    tupleArr = []
    for item in sortedList:
        tuple = (item[0], item[1], hash[item[0]])
        tupleArr.append(tuple)

    return tupleArr


def sortHash(hash, sortField, reversedFlag):
    if hash is None:
        return None
    sortedList = sorted(hash.items(), key = lambda a:a[sortField], reverse=reversedFlag)
    return sortedList

def output_sortedHash(hash, sortField, reversedFlag):
    sortedList = sortHash(hash, sortField, reversedFlag)
    for item in sortedList:
        print item[0], "\t", item[1]


# appHash: key:app
# app: val:cumuStep
def updateAppHash(appHash, key, val, cumuStep):
    app = {}
    if key in appHash:
        app = appHash[key]

    cumulativeInsert(app, val, cumuStep)
    appHash[key] = app
    return appHash


# two list contain at least one same item
def hasSameKey_le1_inlist(itemList1, itemList2):
    commonList = [item for item in itemList1 if item in itemList2]
    return len(commonList)


# statistic num of items whose value in valArr
def statisticHash(hash, valArr):
    numArr = []
    for val in valArr[:-1]:
        itemList = [item for item in hash if hash[item] == val]
        numArr.append(len(itemList))
    numArr.append(len([item for item in hash if hash[item] >= valArr[-1]]))
    return numArr

