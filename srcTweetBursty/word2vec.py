
def loadWord2Vec(modelPath):
    content = open(modelPath, "r").readlines()
    wordNum, dim = content[0].strip().split()
    content = [line.strip().split() for line in content[1:]]
    content = [(item[0], [float(val) for val in item[1:]]) for item in content]
    #word2vecModel = {} #word:vector
    word2vecModel = dict(content)
    return word2vecModel



