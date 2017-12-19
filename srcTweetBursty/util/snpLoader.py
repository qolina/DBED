# load companies in snp500
def loadSnP500(snpfilename):
    companies = open(snpfilename, "r").readlines()
    companies = [line.strip().lower() for line in companies]
    syms = [line.split("\t")[0] for line in companies]
    names = [line.split("\t")[1] for line in companies]
    sym_names = zip(syms, names)
    #print "## End of reading file. [snp500 file][with rank]  snp companies: ", len(syms), "eg:", sym_names[0], snpfilename
    return sym_names

